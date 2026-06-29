from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import smplx
from smplx.lbs import batch_rigid_transform, batch_rodrigues, blend_shapes, vertices2joints
import torch


REQUIRED_OUTPUTS = [
    "stage_a_template_weights.png",
    "stage_b_shaped_joints.png",
    "stage_c_pose_offsets.png",
    "stage_d_lbs_result.png",
    "comparison_grid.png",
    "all_joint_weights.png",
    "summary.txt",
]


class ModelFileMissing(FileNotFoundError):
    def __init__(self, searched_root: str | os.PathLike[str]) -> None:
        super().__init__(
            f"SMPL_NEUTRAL.pkl not found. Put the official file under {searched_root}/models/ "
            "or set SMPL_MODEL_PATH to the file/folder downloaded from the course cloud or SMPL website."
        )


@dataclass
class LbsIntermediates:
    v_template: torch.Tensor
    v_shaped: torch.Tensor
    J: torch.Tensor
    v_posed: torch.Tensor
    verts: torch.Tensor
    J_transformed: torch.Tensor
    pose_offsets: torch.Tensor
    rot_mats: torch.Tensor
    A: torch.Tensor
    betas: torch.Tensor
    body_pose: torch.Tensor
    global_orient: torch.Tensor


def find_model_file(candidates: Iterable[str | os.PathLike[str]] | None = None) -> Path | None:
    roots: list[Path] = []
    if candidates:
        roots.extend(Path(path).expanduser() for path in candidates)
    env_path = os.environ.get("SMPL_MODEL_PATH")
    if env_path:
        roots.append(Path(env_path).expanduser())
    roots.extend(
        [
            Path(__file__).resolve().parents[1] / "models",
            Path.cwd() / "models",
            Path.cwd() / "work8" / "models",
        ]
    )

    names = [
        "SMPL_NEUTRAL.pkl",
        "smpl/SMPL_NEUTRAL.pkl",
        "SMPL_NEUTRAL/SMPL_NEUTRAL.pkl",
    ]
    for root in roots:
        if root.is_file() and root.name == "SMPL_NEUTRAL.pkl":
            return root
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return candidate
    return None


def load_smpl_model(model_file: Path) -> smplx.SMPL:
    return smplx.create(
        str(model_file),
        model_type="smpl",
        gender="neutral",
        ext="pkl",
        batch_size=1,
        create_transl=False,
    )


def make_assignment_parameters(model: smplx.SMPL) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = model.v_template.dtype
    device = model.v_template.device
    num_betas = int(model.shapedirs.shape[-1])
    betas = torch.zeros([1, num_betas], dtype=dtype, device=device)
    values = torch.tensor([1.2, -0.8, 0.55, 0.35, -0.25], dtype=dtype, device=device)
    betas[0, : min(num_betas, values.numel())] = values[: min(num_betas, values.numel())]

    body_pose = torch.zeros([1, 69], dtype=dtype, device=device)
    # Non-zero pose: bend limbs and slightly twist torso. The indices are SMPL body-pose joints
    # after the global/root joint.
    for joint_index, axis, angle in [(2, 0, 0.35), (5, 0, -0.45), (15, 2, -0.85), (18, 2, 0.85), (20, 1, -0.65)]:
        base = joint_index * 3
        if base + axis < body_pose.shape[1]:
            body_pose[0, base + axis] = angle

    global_orient = torch.zeros([1, 3], dtype=dtype, device=device)
    return betas, global_orient, body_pose


def manual_lbs(model: smplx.SMPL, betas: torch.Tensor, global_orient: torch.Tensor, body_pose: torch.Tensor) -> LbsIntermediates:
    pose = torch.cat([global_orient, body_pose], dim=1)
    batch_size = betas.shape[0]
    dtype = betas.dtype
    device = betas.device

    v_template = model.v_template.unsqueeze(0).expand(batch_size, -1, -1)
    v_shaped = v_template + blend_shapes(betas, model.shapedirs)
    J = vertices2joints(model.J_regressor, v_shaped)

    ident = torch.eye(3, dtype=dtype, device=device)
    rot_mats = batch_rodrigues(pose.reshape(-1, 3)).reshape(batch_size, -1, 3, 3)
    pose_feature = (rot_mats[:, 1:, :, :] - ident).reshape(batch_size, -1)
    pose_offsets = torch.matmul(pose_feature, model.posedirs).reshape(batch_size, -1, 3)
    v_posed = v_shaped + pose_offsets

    J_transformed, A = batch_rigid_transform(rot_mats, J, model.parents, dtype=dtype)
    weights = model.lbs_weights.unsqueeze(0).expand(batch_size, -1, -1)
    num_joints = model.J_regressor.shape[0]
    transform = torch.matmul(weights, A.reshape(batch_size, num_joints, 16)).reshape(batch_size, -1, 4, 4)
    v_posed_homo = torch.cat(
        [v_posed, torch.ones([batch_size, v_posed.shape[1], 1], dtype=dtype, device=device)],
        dim=2,
    )
    verts = torch.matmul(transform, v_posed_homo.unsqueeze(-1))[:, :, :3, 0]

    return LbsIntermediates(
        v_template=v_template,
        v_shaped=v_shaped,
        J=J,
        v_posed=v_posed,
        verts=verts,
        J_transformed=J_transformed,
        pose_offsets=pose_offsets,
        rot_mats=rot_mats,
        A=A,
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient,
    )


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def front_projection(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return vertices[:, 0], vertices[:, 1]


def plot_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_path: Path,
    title: str,
    values: np.ndarray | None = None,
    joints: np.ndarray | None = None,
    cmap: str = "viridis",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x, y = front_projection(vertices)
    triangulation = mtri.Triangulation(x, y, triangles=faces)

    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=160)
    ax.set_title(title, fontsize=12)
    if values is None:
        ax.triplot(triangulation, color="#355c7d", linewidth=0.18, alpha=0.55)
        ax.tripcolor(triangulation, np.full(vertices.shape[0], 0.55), cmap="Blues", shading="gouraud", alpha=0.55)
    else:
        mesh = ax.tripcolor(triangulation, values, cmap=cmap, shading="gouraud")
        fig.colorbar(mesh, ax=ax, shrink=0.72)
    if joints is not None:
        ax.scatter(joints[:, 0], joints[:, 1], s=22, c="#e43d30", edgecolors="white", linewidths=0.5, zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def make_comparison_grid(output_dir: Path) -> None:
    labels = [
        ("(a) template + weights", "stage_a_template_weights.png"),
        ("(b) shape + joints", "stage_b_shaped_joints.png"),
        ("(c) pose offsets", "stage_c_pose_offsets.png"),
        ("(d) final skinned mesh", "stage_d_lbs_result.png"),
    ]
    thumbs = []
    for label, filename in labels:
        image = Image.open(output_dir / filename).convert("RGB").resize((420, 420))
        canvas = Image.new("RGB", (420, 456), "white")
        canvas.paste(image, (0, 36))
        draw = ImageDraw.Draw(canvas)
        try:
            title_font = ImageFont.truetype("Arial.ttf", 18)
        except OSError:
            title_font = ImageFont.load_default()
        draw.text((14, 8), label, fill="#182230", font=title_font)
        thumbs.append(canvas)
    grid = Image.new("RGB", (840, 912), "white")
    grid.paste(thumbs[0], (0, 0))
    grid.paste(thumbs[1], (420, 0))
    grid.paste(thumbs[2], (0, 456))
    grid.paste(thumbs[3], (420, 456))
    grid.save(output_dir / "comparison_grid.png")


def make_animation(output_dir: Path, model: smplx.SMPL, base_betas: torch.Tensor) -> None:
    frames = []
    body_pose = torch.zeros([1, 69], dtype=base_betas.dtype, device=base_betas.device)
    global_orient = torch.zeros([1, 3], dtype=base_betas.dtype, device=base_betas.device)
    faces = np.asarray(model.faces, dtype=np.int32)
    for frame in range(32):
        t = frame / 31
        body_pose.zero_()
        body_pose[0, 15 * 3 + 2] = -0.95 * t
        inter = manual_lbs(model, base_betas, global_orient, body_pose)
        image_path = output_dir / "_animation_frame.png"
        weights = tensor_to_numpy(model.lbs_weights[:, 16])
        plot_mesh(tensor_to_numpy(inter.verts[0]), faces, image_path, f"SMPL LBS animation frame {frame:02d}", values=weights)
        frames.append(Image.open(image_path).convert("RGB").resize((540, 540)))
    (output_dir / "_animation_frame.png").unlink(missing_ok=True)
    frames[0].save(output_dir / "lbs_pose_animation.gif", save_all=True, append_images=frames[1:], duration=65, loop=0, optimize=True)


def write_summary(
    output_dir: Path,
    model: smplx.SMPL,
    model_file: Path,
    inter: LbsIntermediates,
    official_vertices: torch.Tensor,
) -> None:
    diff = torch.abs(inter.verts - official_vertices)
    mean_abs_error = float(diff.mean().detach().cpu())
    max_abs_error = float(diff.max().detach().cpu())
    summary = output_dir / "summary.txt"
    summary.write_text(
        "\n".join(
            [
                "Work8 SMPL LBS summary",
                "Author: 许艺珈",
                f"model_file: {model_file}",
                f"vertices: {int(model.v_template.shape[0])}",
                f"faces: {int(model.faces.shape[0])}",
                f"joints: {int(model.J_regressor.shape[0])}",
                f"betas_dim: {int(model.shapedirs.shape[-1])}",
                "core_objects:",
                f"  v_template: {tuple(inter.v_template.shape)}",
                f"  v_shaped: {tuple(inter.v_shaped.shape)}",
                f"  J: {tuple(inter.J.shape)}",
                f"  v_posed: {tuple(inter.v_posed.shape)}",
                f"  verts: {tuple(inter.verts.shape)}",
                "manual_vs_official:",
                f"  mean_absolute_error: {mean_abs_error:.10f}",
                f"  max_absolute_error: {max_abs_error:.10f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_pipeline(output_dir: str | os.PathLike[str], model_candidates: Iterable[str | os.PathLike[str]] | None = None) -> Path:
    work_root = Path(__file__).resolve().parents[1]
    model_file = find_model_file(model_candidates)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if model_file is None:
        (output_path / "MISSING_MODEL.txt").write_text(
            "SMPL_NEUTRAL.pkl not found. Place the official file in work8/models/SMPL_NEUTRAL.pkl "
            "or set SMPL_MODEL_PATH, then run `python3 main.py` again.\n",
            encoding="utf-8",
        )
        raise ModelFileMissing(work_root)

    model = load_smpl_model(model_file)
    model.eval()
    betas, global_orient, body_pose = make_assignment_parameters(model)
    inter = manual_lbs(model, betas, global_orient, body_pose)
    official = model.forward(
        betas=betas,
        global_orient=global_orient,
        body_pose=body_pose,
        return_verts=True,
    )
    official_vertices = official.vertices

    faces = np.asarray(model.faces, dtype=np.int32)
    template = tensor_to_numpy(inter.v_template[0])
    shaped = tensor_to_numpy(inter.v_shaped[0])
    posed = tensor_to_numpy(inter.v_posed[0])
    verts = tensor_to_numpy(inter.verts[0])
    joints = tensor_to_numpy(inter.J[0])
    transformed_joints = tensor_to_numpy(inter.J_transformed[0])
    lbs_weights = tensor_to_numpy(model.lbs_weights)
    chosen_joint = min(16, lbs_weights.shape[1] - 1)
    joint_weights = lbs_weights[:, chosen_joint]
    dominant_joint = lbs_weights.argmax(axis=1)
    dominant_strength = lbs_weights.max(axis=1)
    pose_offset_norm = np.linalg.norm(tensor_to_numpy(inter.pose_offsets[0]), axis=1)

    plot_mesh(
        template,
        faces,
        output_path / "stage_a_template_weights.png",
        f"(a) template + joint {chosen_joint} weights",
        values=joint_weights,
    )
    plot_mesh(
        template,
        faces,
        output_path / "all_joint_weights.png",
        "optional: dominant joint weights",
        values=dominant_joint + dominant_strength,
        cmap="turbo",
    )
    plot_mesh(
        shaped,
        faces,
        output_path / "stage_b_shaped_joints.png",
        "(b) v_shaped + regressed joints J(beta)",
        joints=joints,
    )
    plot_mesh(
        posed,
        faces,
        output_path / "stage_c_pose_offsets.png",
        "(c) pose corrective magnitude ||B_P(theta)||",
        values=pose_offset_norm,
        cmap="magma",
    )
    plot_mesh(
        verts,
        faces,
        output_path / "stage_d_lbs_result.png",
        "(d) final verts after Linear Blend Skinning",
        joints=transformed_joints,
    )
    make_comparison_grid(output_path)
    make_animation(output_path, model, betas)
    write_summary(output_path, model, model_file, inter, official_vertices)
    return output_path
