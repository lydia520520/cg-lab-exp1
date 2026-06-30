from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np
import taichi as ti


WIDTH = 900
HEIGHT = 760
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1600
POINT_RADIUS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Work3 optional: anti-aliased Bezier and B-spline visualizer.")
    parser.add_argument("--arch", choices=["cpu", "gpu"], default="cpu", help="Taichi backend.")
    parser.add_argument("--headless", action="store_true", help="Run without opening the GUI window.")
    parser.add_argument("--max-frames", type=int, default=0, help="Exit after N frames in automated tests. 0 means no limit.")
    return parser.parse_args()


def init_taichi(arch: str) -> None:
    ti.init(arch=ti.cpu if arch == "cpu" else ti.gpu)


def de_casteljau(points: list[np.ndarray], t: float) -> np.ndarray:
    current = [point.astype(np.float32) for point in points]
    while len(current) > 1:
        current = [(1.0 - t) * current[i] + t * current[i + 1] for i in range(len(current) - 1)]
    return current[0]


def cox_de_boor(i: int, k: int, t: float, knots: np.ndarray) -> float:
    if k == 0:
        if knots[i] <= t < knots[i + 1]:
            return 1.0
        if t == knots[-1] and i + 1 == len(knots) - 1:
            return 1.0
        return 0.0

    left_denom = knots[i + k] - knots[i]
    right_denom = knots[i + k + 1] - knots[i + 1]
    left = 0.0
    right = 0.0
    if left_denom > 1e-6:
        left = (t - knots[i]) / left_denom * cox_de_boor(i, k - 1, t, knots)
    if right_denom > 1e-6:
        right = (knots[i + k + 1] - t) / right_denom * cox_de_boor(i + 1, k - 1, t, knots)
    return left + right


def sample_bezier(control_points: list[np.ndarray]) -> np.ndarray:
    if len(control_points) < 2:
        return np.empty((0, 2), dtype=np.float32)
    samples = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
    for idx in range(NUM_SEGMENTS + 1):
        t = idx / NUM_SEGMENTS
        samples[idx] = de_casteljau(control_points, t)
    return samples


def sample_bspline(control_points: list[np.ndarray], degree: int = 3) -> np.ndarray:
    if len(control_points) < degree + 1:
        return np.empty((0, 2), dtype=np.float32)

    points = np.asarray(control_points, dtype=np.float32)
    n = len(points) - 1
    knots = np.concatenate(
        [
            np.zeros(degree, dtype=np.float32),
            np.arange(n - degree + 2, dtype=np.float32),
            np.full(degree, n - degree + 1, dtype=np.float32),
        ]
    )
    t_min = knots[degree]
    t_max = knots[n + 1]

    samples = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
    for idx in range(NUM_SEGMENTS + 1):
        t = t_min + (t_max - t_min) * (idx / NUM_SEGMENTS)
        point = np.zeros(2, dtype=np.float32)
        for i in range(n + 1):
            point += cox_de_boor(i, degree, t, knots) * points[i]
        samples[idx] = point
    return samples


def draw_aa_point(image: np.ndarray, point: np.ndarray, color: np.ndarray, radius: float = 1.25) -> None:
    x = point[0] * (WIDTH - 1)
    y = point[1] * (HEIGHT - 1)
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))

    for dx in range(-1, 2):
        for dy in range(-1, 2):
            px = x0 + dx
            py = y0 + dy
            if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                dist2 = (px + 0.5 - x) ** 2 + (py + 0.5 - y) ** 2
                weight = np.exp(-dist2 / (2.0 * radius * radius))
                image[px, py] = np.maximum(image[px, py], color * weight)


def rasterize_curve(image: np.ndarray, samples: np.ndarray, anti_aliasing: bool) -> None:
    curve_color = np.array([0.10, 0.84, 0.32], dtype=np.float32)
    if anti_aliasing:
        for point in samples:
            draw_aa_point(image, point, curve_color)
    else:
        for point in samples:
            px = int(point[0] * (WIDTH - 1))
            py = int(point[1] * (HEIGHT - 1))
            if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                image[px, py] = curve_color


def draw_control_polygon(gui: ti.GUI, control_points: Iterable[np.ndarray]) -> None:
    points = list(control_points)
    if not points:
        return

    np_points = np.asarray(points, dtype=np.float32)
    gui.circles(np_points, radius=POINT_RADIUS, color=0xF04F5C)
    if len(points) >= 2:
        for idx in range(len(points) - 1):
            gui.line(points[idx], points[idx + 1], radius=1.5, color=0x7D8795)


def default_points() -> list[np.ndarray]:
    return [
        np.array([0.12, 0.78], dtype=np.float32),
        np.array([0.28, 0.18], dtype=np.float32),
        np.array([0.45, 0.84], dtype=np.float32),
        np.array([0.62, 0.26], dtype=np.float32),
        np.array([0.82, 0.70], dtype=np.float32),
    ]


def main() -> None:
    args = parse_args()
    init_taichi(args.arch)

    control_points = default_points() if args.headless else []
    use_bspline = False
    anti_aliasing = True
    frame = 0

    if args.headless:
        while True:
            samples = sample_bspline(control_points) if use_bspline else sample_bezier(control_points)
            image = np.zeros((WIDTH, HEIGHT, 3), dtype=np.float32)
            rasterize_curve(image, samples, anti_aliasing)
            frame += 1
            if args.max_frames and frame >= args.max_frames:
                break
        return

    gui = ti.GUI("Work3 Extra - Anti-Aliasing and B-Spline", res=(WIDTH, HEIGHT), background_color=0xF6F7FB)

    while gui.running:
        if args.max_frames and frame >= args.max_frames:
            break

        for event in gui.get_events(ti.GUI.PRESS):
            if event.key == ti.GUI.LMB and len(control_points) < MAX_CONTROL_POINTS:
                control_points.append(np.array(gui.get_cursor_pos(), dtype=np.float32))
            elif event.key in ("c", "C"):
                control_points = []
            elif event.key in ("b", "B"):
                use_bspline = not use_bspline
            elif event.key in ("a", "A"):
                anti_aliasing = not anti_aliasing
            elif event.key == ti.GUI.ESCAPE:
                gui.running = False

        samples = sample_bspline(control_points) if use_bspline else sample_bezier(control_points)
        image = np.zeros((WIDTH, HEIGHT, 3), dtype=np.float32)
        rasterize_curve(image, samples, anti_aliasing)

        gui.set_image(image)
        draw_control_polygon(gui, control_points)

        mode_text = "B-Spline" if use_bspline else "Bezier"
        aa_text = "On" if anti_aliasing else "Off"
        gui.text("Optional Demo: anti-aliased curve rendering and B-spline toggle", pos=(0.02, 0.02), color=0x223344)
        gui.text(f"Mode: {mode_text}", pos=(0.02, 0.06), color=0x223344)
        gui.text(f"Anti-aliasing: {aa_text}", pos=(0.02, 0.10), color=0x223344)
        gui.text("LMB:add point  B:toggle B-spline  A:toggle AA  C:clear  ESC:quit", pos=(0.02, 0.14), color=0x556677)
        if use_bspline and len(control_points) < 4:
            gui.text("B-spline needs at least 4 control points.", pos=(0.02, 0.18), color=0xA44B4B)

        gui.show()
        frame += 1


if __name__ == "__main__":
    main()
