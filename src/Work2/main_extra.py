from __future__ import annotations

import argparse
import math

import numpy as np
import taichi as ti


CUBE_VERTICES = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)

EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Work2 optional: perspective cube rotation interpolation.")
    parser.add_argument("--arch", choices=["cpu", "gpu"], default="cpu", help="Taichi backend.")
    parser.add_argument("--max-frames", type=int, default=0, help="Exit automatically after N frames. 0 means run until closed.")
    parser.add_argument("--headless", action="store_true", help="Run update logic without opening the GUI window.")
    return parser.parse_args()


def init_taichi(arch: str) -> None:
    backend = ti.cpu if arch == "cpu" else ti.gpu
    ti.init(arch=backend)


def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    return quat / np.linalg.norm(quat)


def quaternion_from_euler(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx = math.radians(rx_deg) * 0.5
    ry = math.radians(ry_deg) * 0.5
    rz = math.radians(rz_deg) * 0.5

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    quat = np.array(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ],
        dtype=np.float32,
    )
    return normalize_quaternion(quat)


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        return normalize_quaternion(q0 + t * (q1 - q0))

    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1


def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quaternion(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def transform_vertices(rotation: np.ndarray, center: np.ndarray, scale: float = 0.8) -> np.ndarray:
    return (CUBE_VERTICES * scale) @ rotation.T + center


def project_points(points: np.ndarray, eye: np.ndarray, fov_deg: float, aspect_ratio: float) -> np.ndarray:
    relative = points - eye
    camera_z = -relative[:, 2]
    focal = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    x_ndc = (relative[:, 0] * focal / aspect_ratio) / camera_z
    y_ndc = (relative[:, 1] * focal) / camera_z

    projected = np.stack([(x_ndc + 1.0) * 0.5, (y_ndc + 1.0) * 0.5], axis=1)
    projected[:, 1] = 1.0 - projected[:, 1]
    return projected.astype(np.float32)


def draw_cube(gui: ti.GUI, vertices: np.ndarray, color: int, radius: float = 2.2) -> None:
    for start, end in EDGES:
        gui.line(vertices[start], vertices[end], radius=radius, color=color)
    gui.circles(vertices, radius=4.0, color=color)


def main() -> None:
    args = parse_args()
    init_taichi(args.arch)

    eye = np.array([0.0, 0.0, 5.5], dtype=np.float32)
    aspect_ratio = 900.0 / 700.0

    q_start = quaternion_from_euler(0.0, 0.0, 0.0)
    q_end = quaternion_from_euler(35.0, 125.0, -20.0)

    center_left = np.array([-2.7, -0.5, -1.8], dtype=np.float32)
    center_mid = np.array([0.0, -0.05, -2.1], dtype=np.float32)
    center_right = np.array([2.7, -0.5, -1.8], dtype=np.float32)

    frame = 0
    paused = False
    phase = 0.0

    gui = None if args.headless else ti.GUI("Work2 Extra - Cube Rotation Interpolation", res=(900, 700), background_color=0xF7F8FC)

    while True:
        if gui is not None and not gui.running:
            break
        if args.max_frames and frame >= args.max_frames:
            break

        if gui is not None:
            while gui.get_event(ti.GUI.PRESS):
                if gui.event.key == ti.GUI.ESCAPE:
                    gui.running = False
                elif gui.event.key == ti.GUI.SPACE:
                    paused = not paused
                elif gui.event.key in ("r", "R"):
                    phase = 0.0

        if not paused:
            phase += 0.02

        alpha = 0.5 * (1.0 + math.sin(phase))
        q_mid = slerp(q_start, q_end, alpha)

        left_proj = project_points(transform_vertices(quaternion_to_matrix(q_start), center_left), eye, 45.0, aspect_ratio)
        right_proj = project_points(transform_vertices(quaternion_to_matrix(q_end), center_right), eye, 45.0, aspect_ratio)
        mid_proj = project_points(transform_vertices(quaternion_to_matrix(q_mid), center_mid), eye, 45.0, aspect_ratio)

        if gui is not None:
            draw_cube(gui, left_proj, color=0x85BDF2, radius=1.7)
            draw_cube(gui, right_proj, color=0x85BDF2, radius=1.7)
            draw_cube(gui, mid_proj, color=0xF39A4A, radius=2.5)

            gui.text("Optional: 3D cube with perspective projection and rotation interpolation", pos=(0.03, 0.03), color=0x223344)
            gui.text("Left: R0   Middle: Rt   Right: R1", pos=(0.03, 0.07), color=0x445566)
            gui.text("SPACE: pause/resume   R: reset   ESC: quit", pos=(0.03, 0.11), color=0x556677)
            gui.text(f"alpha = {alpha:.2f}", pos=(0.03, 0.15), color=0x8A4B16)
            gui.show()

        frame += 1


if __name__ == "__main__":
    main()
