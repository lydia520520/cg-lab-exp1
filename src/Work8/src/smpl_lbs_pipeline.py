from __future__ import annotations

import os
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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


class _ChumpyArrayShim:
    """Minimal shim so legacy SMPL pickles can be loaded without installing chumpy."""

    def __setstate__(self, state: object) -> None:
        self.__dict__.update(state)

    def _array(self) -> np.ndarray:
        if hasattr(self, "r"):
            return self.r
        if hasattr(self, "x"):
            return self.x
        raise AttributeError("Cannot recover array data from chumpy pickle object")

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        return np.asarray(self._array(), dtype=dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return np.asarray(self).shape

    def __len__(self) -> int:
        return len(np.asarray(self))

    def __getitem__(self, item: object) -> np.ndarray:
        return np.asarray(self)[item]


def install_chumpy_pickle_shim() -> None:
    if "chumpy.ch" in sys.modules:
        return

    chumpy_module = types.ModuleType("chumpy")
    chumpy_ch_module = types.ModuleType("chumpy.ch")
    _ChumpyArrayShim.__name__ = "Ch"
    _ChumpyArrayShim.__qualname__ = "Ch"
    _ChumpyArrayShim.__module__ = "chumpy.ch"
    chumpy_ch_module.Ch = _ChumpyArrayShim
    chumpy_module.ch = chumpy_ch_module

    sys.modules["chumpy"] = chumpy_module
    sys.modules["chumpy.ch"] = chumpy_ch_module


class ModelFileMissing(FileNotFoundError):
    def __init__(self, searched_root: str | os.PathLike[str]) -> None:
        super().__init__(
            f"SMPL_NEUTRAL.pkl not found. Put the official file under {searched_root}/models/ "
            "or place it in the Work8 root directory, or set SMPL_MODEL_PATH."
        )


@dataclass
class LbsIntermediates:
    v_template: torch.Tensor
    J_template: torch.Tensor
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

    work_root = Path(__file__).resolve().parents[1]
    roots.extend(
        [
            work_root,
            work_root / "models",
            Path.cwd(),
            Path.cwd() / "models",
            Path.cwd() / "work8",
            Path.cwd() / "work8" / "models",
            Path.cwd() / "Work8",
            Path.cwd() / "Work8" / "models",
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
    install_chumpy_pickle_shim()
    model_dir = prepare_model_root_for_smplx(model_file)
    return smplx.create(
        model_path=str(model_dir),
        model_type="smpl",
        gender="neutral",
        ext="pkl",
        batch_size=1,
        create_transl=False,
    )


def prepare_model_root_for_smplx(model_file: Path) -> Path:
    if model_file.is_dir():
        return model_file

    if model_file.parent.name == "smpl":
        return model_file.parent.parent

    model_root = model_file.parent
    smpl_dir = model_root / "smpl"
    smpl_dir.mkdir(parents=True, exist_ok=True)
    target_file = smpl_dir / "SMPL_NEUTRAL.pkl"
    if not target_file.exists():
        try:
            target_file.symlink_to(model_file)
        except OSError:
            shutil.copy2(model_file, target_file)
    return model_root


def make_assignment_parameters(model: smplx.SMPL) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = model.v_template.dtype
    device = model.v_template.device
    num_betas = int(model.shapedirs.shape[-1])

    betas = torch.zeros((1, num_betas), dtype=dtype, device=device)
    demo_values = torch.tensor([1.2, -0.8, 0.55, 0.35, -0.25], dtype=dtype, device=device)
    betas[0, : min(num_betas, demo_values.numel())] = demo_values[: min(num_betas, demo_values.numel())]

    body_pose = torch.zeros((1, 69), dtype=dtype, device=device)
    for joint_index, axis, angle in [(2, 0, 0.35), (5, 0, -0.45), (15, 2, -0.85), (18, 2, 0.85), (20, 1, -0.65)]:
        base = joint_index * 3
        if base + axis < body_pose.shape[1]:
            body_pose[0, base + axis] = angle

    global_orient = torch.zeros((1, 3), dtype=dtype, device=device)
    return betas, global_orient, body_pose


def prepare_posedirs(posedirs: torch.Tensor, expected_pose_dim: int) -> torch.Tensor:
    if posedirs.dim() != 2:
        posedirs = posedirs.reshape(posedirs.shape[0], -1)
    if posedirs.shape[0] == expected_pose_dim:
        return posedirs
    if posedirs.shape[1] == expected_pose_dim:
        return posedirs.T
    raise RuntimeError(
        f"posedirs shape mismatch: posedirs.shape={tuple(posedirs.shape)}, expected_pose_dim={expected_pose_dim}"
    )


def manual_lbs(model: smplx.SMPL, betas: torch.Tensor, global_orient: torch.Tensor, body_pose: torch.Tensor) -> LbsIntermediates:
    pose = torch.cat([global_orient, body_pose], dim=1)
    batch_size = betas.shape[0]
    dtype = betas.dtype
    device = betas.device

    v_template = model.v_template
    if v_template.dim() == 2:
        v_template = v_template.unsqueeze(0)
    v_template = v_template.expand(batch_size, -1, -1)
    J_template = vertices2joints(model.J_regressor, v_template)

    shapedirs = model.shapedirs[:, :, : betas.shape[1]]
    v_shaped = v_template + blend_shapes(betas, shapedirs)
    J = vertices2joints(model.J_regressor, v_shaped)

    ident = torch.eye(3, dtype=dtype, device=device)
    rot_mats = batch_rodrigues(pose.reshape(-1, 3)).reshape(batch_size, -1, 3, 3)
    pose_feature = (rot_mats[:, 1:, :, :] - ident).reshape(batch_size, -1)
    posedirs = prepare_posedirs(model.posedirs, expected_pose_dim=pose_feature.shape[1])
    pose_offsets = torch.matmul(pose_feature, posedirs).reshape(batch_size, -1, 3)
    v_posed = v_shaped + pose_offsets

    J_transformed, A = batch_rigid_transform(rot_mats, J, model.parents, dtype=dtype)
    num_joints = J.shape[1]
    weights = model.lbs_weights.unsqueeze(0).expand(batch_size, -1, -1)
    transform = torch.matmul(weights, A.reshape(batch_size, num_joints, 16)).reshape(batch_size, -1, 4, 4)
    v_posed_homo = torch.cat(
        [v_posed, torch.ones((batch_size, v_posed.shape[1], 1), dtype=dtype, device=device)],
        dim=2,
    )
    verts = torch.matmul(transform, v_posed_homo.unsqueeze(-1))[:, :, :3, 0]

    return LbsIntermediates(
        v_template=v_template,
        J_template=J_template,
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


def to_plot_coords(points: np.ndarray) -> np.ndarray:
    return points[:, [0, 2, 1]]


def get_face_colors_from_vertex_scalar(vertex_scalar: np.ndarray, faces: np.ndarray, cmap_name: str = "viridis") -> np.ndarray:
    scalar = vertex_scalar.astype(np.float64)
    scalar = (scalar - scalar.min()) / (scalar.max() - scalar.min() + 1e-8)
    face_scalar = scalar[faces].mean(axis=1)
    cmap = plt.get_cmap(cmap_name)
    return cmap(face_scalar)


def get_face_colors_from_joint_weights(lbs_weights: np.ndarray, faces: np.ndarray) -> np.ndarray:
    face_weights = lbs_weights[faces].mean(axis=1)
    dominant_joint = np.argmax(face_weights, axis=1)
    dominant_weight = np.max(face_weights, axis=1)
    palette = plt.get_cmap("hsv")(np.linspace(0.0, 1.0, lbs_weights.shape[1], endpoint=False))
    face_colors = palette[dominant_joint]
    strength = 0.35 + 0.65 * dominant_weight
    face_colors[:, :3] = face_colors[:, :3] * strength[:, None] + (1.0 - strength[:, None]) * 0.88
    face_colors[:, 3] = 1.0
    return face_colors


def set_axes_equal(ax: object, vertices: np.ndarray) -> None:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = 0.5 * np.max(maxs - mins + 1e-8)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def shade_face_colors(vertices: np.ndarray, faces: np.ndarray, face_colors: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8

    light_dir = np.array([-0.25, -0.55, 0.80], dtype=np.float64)
    light_dir /= np.linalg.norm(light_dir)
    intensity = 0.35 + 0.65 * np.clip(normals @ light_dir, 0.0, 1.0)

    shaded = face_colors.copy()
    shaded[:, :3] *= intensity[:, None]
    return shaded


def draw_mesh(
    ax: object,
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray | None = None,
    vertex_scalar: np.ndarray | None = None,
    face_colors: np.ndarray | None = None,
    title: str = "",
    elev: float = 12.0,
    azim: float = 108.0,
) -> None:
    plot_vertices = to_plot_coords(vertices)
    plot_joints = None if joints is None else to_plot_coords(joints)

    if face_colors is not None:
        colors = face_colors.copy()
    elif vertex_scalar is None:
        colors = np.tile(np.array([[0.82, 0.67, 0.52, 1.0]]), (faces.shape[0], 1))
    else:
        colors = get_face_colors_from_vertex_scalar(vertex_scalar, faces)
    colors = shade_face_colors(plot_vertices, faces, colors)

    mesh = Poly3DCollection(
        plot_vertices[faces],
        facecolors=colors,
        linewidths=0.03,
        edgecolors=(0.0, 0.0, 0.0, 0.05),
    )
    ax.add_collection3d(mesh)

    if plot_joints is not None:
        ax.scatter(
            plot_joints[:, 0],
            plot_joints[:, 1],
            plot_joints[:, 2],
            c="white",
            s=12,
            depthshade=False,
            edgecolors="black",
            linewidths=0.3,
        )

    set_axes_equal(ax, plot_vertices)
    ax.set_proj_type("persp", focal_length=0.85)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)


def plot_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_path: Path,
    title: str,
    values: np.ndarray | None = None,
    joints: np.ndarray | None = None,
    cmap: str = "viridis",
    face_colors: np.ndarray | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(5, 6), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    if values is not None and face_colors is None:
        face_colors = get_face_colors_from_vertex_scalar(values, faces, cmap_name=cmap)
    draw_mesh(ax, vertices, faces, joints=joints, face_colors=face_colors, title=title)
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
    frames: list[Image.Image] = []
    body_pose = torch.zeros((1, 69), dtype=base_betas.dtype, device=base_betas.device)
    global_orient = torch.zeros((1, 3), dtype=base_betas.dtype, device=base_betas.device)
    faces = np.asarray(model.faces, dtype=np.int32)
    joint_index = min(16, model.lbs_weights.shape[1] - 1)
    weights = tensor_to_numpy(model.lbs_weights[:, joint_index])

    for frame in range(24):
        t = frame / 23.0
        body_pose.zero_()
        body_pose[0, 15 * 3 + 2] = -0.95 * t
        inter = manual_lbs(model, base_betas, global_orient, body_pose)
        image_path = output_dir / "_animation_frame.png"
        plot_mesh(
            tensor_to_numpy(inter.verts[0]),
            faces,
            image_path,
            f"SMPL LBS animation frame {frame:02d}",
            values=weights,
        )
        frames.append(Image.open(image_path).convert("RGB").resize((540, 540)))

    (output_dir / "_animation_frame.png").unlink(missing_ok=True)
    frames[0].save(
        output_dir / "lbs_pose_animation.gif",
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
        optimize=True,
    )


def show_preview_window(output_dir: Path) -> None:
    fig = plt.figure(figsize=(11.5, 12.8), dpi=140)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.15], hspace=0.14, wspace=0.08)
    panels = [
        ("(a) Template + Weights", output_dir / "stage_a_template_weights.png", grid[0, 0]),
        ("(b) Shape + Joints", output_dir / "stage_b_shaped_joints.png", grid[0, 1]),
        ("(c) Pose Blend Shapes", output_dir / "stage_c_pose_offsets.png", grid[1, 0]),
        ("(d) Final LBS Result", output_dir / "stage_d_lbs_result.png", grid[1, 1]),
        ("All Joint LBS Weights", output_dir / "all_joint_weights.png", grid[2, :]),
    ]

    for title, image_path, spec in panels:
        ax = fig.add_subplot(spec)
        ax.imshow(np.asarray(Image.open(image_path).convert("RGB")))
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle("SMPL LBS Visualization", fontsize=16, y=0.985)
    plt.show()


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


def run_pipeline(
    output_dir: str | os.PathLike[str],
    model_candidates: Iterable[str | os.PathLike[str]] | None = None,
    show_window: bool = True,
) -> Path:
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
    template_joints = tensor_to_numpy(inter.J_template[0])
    shaped = tensor_to_numpy(inter.v_shaped[0])
    posed = tensor_to_numpy(inter.v_posed[0])
    verts = tensor_to_numpy(inter.verts[0])
    joints = tensor_to_numpy(inter.J[0])
    transformed_joints = tensor_to_numpy(inter.J_transformed[0])
    lbs_weights = tensor_to_numpy(model.lbs_weights)
    chosen_joint = min(16, lbs_weights.shape[1] - 1)
    joint_weights = lbs_weights[:, chosen_joint]
    pose_offset_norm = np.linalg.norm(tensor_to_numpy(inter.pose_offsets[0]), axis=1)

    plot_mesh(
        template,
        faces,
        output_path / "stage_a_template_weights.png",
        f"(a) template + joint {chosen_joint} weights",
        values=joint_weights,
        joints=template_joints,
    )
    plot_mesh(
        template,
        faces,
        output_path / "all_joint_weights.png",
        "optional: dominant joint weights",
        joints=template_joints,
        face_colors=get_face_colors_from_joint_weights(lbs_weights, faces),
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
        joints=joints,
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

    if show_window:
        try:
            show_preview_window(output_path)
        except Exception as exc:
            print(f"preview window unavailable: {exc}")
            print(f"saved outputs under: {output_path}")

    return output_path
