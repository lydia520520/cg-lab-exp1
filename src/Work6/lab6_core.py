import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import taichi as ti

DEFAULT_TARGET_LIGHT = (0.8, 0.8, 2.0)
DEFAULT_INITIAL_LIGHT = (0.2, 0.2, -1.2)


@dataclass
class OptimizationRecord:
    step: int
    loss: float
    light_x: float
    light_y: float
    light_z: float
    grad_x: float
    grad_y: float
    grad_z: float
    grad_norm: float


def init_taichi(arch_name: str) -> None:
    arch_map = {
        "cpu": ti.cpu,
        "gpu": ti.gpu,
    }
    if arch_name not in arch_map:
        raise ValueError(f"Unsupported arch: {arch_name}")
    ti.init(arch=arch_map[arch_name], default_fp=ti.f32)


@ti.data_oriented
class SphereLightOptimizer:
    def __init__(
        self,
        width: int,
        height: int,
        target_light: tuple[float, float, float],
        initial_light: tuple[float, float, float],
        sphere_radius: float = 0.75,
        camera_z: float = 2.5,
    ) -> None:
        self.width = width
        self.height = height
        self.aspect = width / height
        self.inv_num_pixels = 1.0 / float(width * height)
        self.sphere_radius = sphere_radius
        self.camera_z = camera_z

        self.target_image = ti.field(dtype=ti.f32, shape=(height, width))
        self.pred_image = ti.field(dtype=ti.f32, shape=(height, width))
        self.loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.light = ti.Vector.field(3, dtype=ti.f32, shape=(), needs_grad=True)
        self.target_light = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.set_target_light(*target_light)
        self.set_light(*initial_light)
        self.render_target_image()
        self.adam_m = np.zeros(3, dtype=np.float32)
        self.adam_v = np.zeros(3, dtype=np.float32)
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_eps = 1e-8

    @ti.kernel
    def set_light(self, x: ti.f32, y: ti.f32, z: ti.f32):
        self.light[None] = ti.Vector([x, y, z])

    @ti.kernel
    def set_target_light(self, x: ti.f32, y: ti.f32, z: ti.f32):
        self.target_light[None] = ti.Vector([x, y, z])

    @ti.func
    def pixel_to_view(self, x: ti.i32, y: ti.i32):
        px = ((ti.cast(x, ti.f32) + 0.5) / self.width) * 2.0 - 1.0
        py = 1.0 - ((ti.cast(y, ti.f32) + 0.5) / self.height) * 2.0
        px *= self.aspect
        return ti.Vector([px, py])

    @ti.func
    def shade_pixel(self, x: ti.i32, y: ti.i32, light: ti.types.vector(3, ti.f32), leak_alpha: ti.f32):
        uv = self.pixel_to_view(x, y)
        rr = self.sphere_radius * self.sphere_radius
        xy_sq = uv.dot(uv)
        value = 0.0
        if xy_sq <= rr:
            z = ti.sqrt(rr - xy_sq)
            hit = ti.Vector([uv.x, uv.y, z])
            normal = hit / self.sphere_radius
            light_dir = light - hit
            light_dir = light_dir / (light_dir.norm() + 1e-6)
            ndotl = normal.dot(light_dir)
            value = ti.max(leak_alpha * ndotl, ndotl)
        return value

    @ti.kernel
    def render_target_image(self):
        for y, x in self.target_image:
            self.target_image[y, x] = self.shade_pixel(x, y, self.target_light[None], 0.0)

    @ti.kernel
    def render_prediction_and_loss(self, leak_alpha: ti.f32):
        for y, x in self.pred_image:
            pred = self.shade_pixel(x, y, self.light[None], leak_alpha)
            self.pred_image[y, x] = pred
            diff = pred - self.target_image[y, x]
            self.loss[None] += diff * diff * self.inv_num_pixels

    @ti.kernel
    def render_prediction_only(self, leak_alpha: ti.f32):
        for y, x in self.pred_image:
            self.pred_image[y, x] = self.shade_pixel(x, y, self.light[None], leak_alpha)

    @ti.kernel
    def apply_gradient_descent(self, lr: ti.f32, limit: ti.f32):
        updated = self.light[None] - lr * self.light.grad[None]
        for i in ti.static(range(3)):
            updated[i] = ti.math.clamp(updated[i], -limit, limit)
        self.light[None] = updated

    def get_light_numpy(self) -> np.ndarray:
        return np.array(self.light[None], dtype=np.float32)

    def get_light_grad_numpy(self) -> np.ndarray:
        return np.array(self.light.grad[None], dtype=np.float32)

    def get_target_image(self) -> np.ndarray:
        return self.target_image.to_numpy()

    def get_pred_image(self) -> np.ndarray:
        return self.pred_image.to_numpy()

    def step(
        self,
        step: int,
        lr: float,
        leak_alpha: float,
        light_limit: float = 4.0,
    ) -> OptimizationRecord:
        with ti.ad.Tape(loss=self.loss):
            self.render_prediction_and_loss(leak_alpha)
        light = self.get_light_numpy()
        grad = self.get_light_grad_numpy()
        loss = float(self.loss[None])
        grad_norm = float(np.linalg.norm(grad))
        record = OptimizationRecord(
            step=step,
            loss=loss,
            light_x=float(light[0]),
            light_y=float(light[1]),
            light_z=float(light[2]),
            grad_x=float(grad[0]),
            grad_y=float(grad[1]),
            grad_z=float(grad[2]),
            grad_norm=grad_norm,
        )
        # Adam keeps the optimization moving even when the raw gradients are tiny.
        self.adam_m = self.adam_beta1 * self.adam_m + (1.0 - self.adam_beta1) * grad
        self.adam_v = self.adam_beta2 * self.adam_v + (1.0 - self.adam_beta2) * (grad * grad)
        m_hat = self.adam_m / (1.0 - self.adam_beta1 ** (step + 1))
        v_hat = self.adam_v / (1.0 - self.adam_beta2 ** (step + 1))
        updated = light - lr * m_hat / (np.sqrt(v_hat) + self.adam_eps)
        updated = np.clip(updated, -light_limit, light_limit)
        self.set_light(float(updated[0]), float(updated[1]), float(updated[2]))
        return record

    def optimize(self, iterations: int, lr: float, leak_alpha: float, light_limit: float = 4.0) -> list[OptimizationRecord]:
        history: list[OptimizationRecord] = []
        for step in range(iterations):
            history.append(self.step(step, lr, leak_alpha, light_limit))
        self.render_prediction_only(leak_alpha)
        return history


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0)


def compose_display_image(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    separator = np.full((left.shape[0], 2), 0.15, dtype=np.float32)
    joined = np.concatenate([left, separator, right], axis=1)
    rgb = np.repeat(joined[..., None], 3, axis=2)
    # Taichi GUI expects image shape to match res=(width, height).
    return np.clip(np.transpose(rgb, (1, 0, 2)), 0.0, 1.0)


def save_pgm(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_8bit = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    height, width = image_8bit.shape
    with path.open('wb') as f:
        f.write(f'P5\n{width} {height}\n255\n'.encode('ascii'))
        f.write(image_8bit.tobytes())


def write_history_csv(path: Path, history: Iterable[OptimizationRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'step', 'loss', 'light_x', 'light_y', 'light_z',
            'grad_x', 'grad_y', 'grad_z', 'grad_norm',
        ])
        for item in history:
            writer.writerow([
                item.step,
                item.loss,
                item.light_x,
                item.light_y,
                item.light_z,
                item.grad_x,
                item.grad_y,
                item.grad_z,
                item.grad_norm,
            ])


def format_history_summary(history: list[OptimizationRecord]) -> str:
    if not history:
        return 'No optimization steps were executed.'
    first = history[0]
    last = history[-1]
    return (
        f'initial_loss={first.loss:.6f}\n'
        f'final_loss={last.loss:.6f}\n'
        f'initial_light=({first.light_x:.4f}, {first.light_y:.4f}, {first.light_z:.4f})\n'
        f'final_light=({last.light_x:.4f}, {last.light_y:.4f}, {last.light_z:.4f})\n'
        f'final_grad_norm={last.grad_norm:.6f}\n'
    )
