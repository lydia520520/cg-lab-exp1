from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import taichi as ti


_TAICHI_READY = False


def ensure_taichi() -> None:
    global _TAICHI_READY
    if not _TAICHI_READY:
        ti.init(arch=ti.cpu, default_fp=ti.f32, offline_cache=True)
        _TAICHI_READY = True


SpringMode = Literal["structural", "shear", "bending"]
Integrator = Literal["explicit", "semi_implicit", "implicit"]


@dataclass(frozen=True)
class SpringCounts:
    structural: int
    shear: int
    bending: int


@ti.data_oriented
class MassSpringCloth:
    def __init__(
        self,
        cols: int = 20,
        rows: int = 20,
        spacing: float = 0.045,
        mass: float = 1.0,
        stiffness: float = 640.0,
        damping: float = 3.0,
        max_speed: float = 3.8,
        enable_shear: bool = True,
        enable_bending: bool = True,
        sphere_center: tuple[float, float, float] = (0.5, 0.05, 0.0),
        sphere_radius: float = 0.16,
    ) -> None:
        ensure_taichi()
        if cols < 2 or rows < 2:
            raise ValueError("cols and rows must both be at least 2")

        self.cols = cols
        self.rows = rows
        self.n_particles = cols * rows
        self.spacing = float(spacing)
        self.mass = float(mass)
        self.stiffness = float(stiffness)
        self.damping = float(damping)
        self.max_speed = float(max_speed)
        self.enable_shear = bool(enable_shear)
        self.enable_bending = bool(enable_bending)
        self.sphere_center_np = np.array(sphere_center, dtype=np.float32)
        self.sphere_radius = float(sphere_radius)

        structural = rows * (cols - 1) + (rows - 1) * cols
        shear = 2 * (rows - 1) * (cols - 1) if enable_shear else 0
        bending = (rows * max(0, cols - 2) + max(0, rows - 2) * cols) if enable_bending else 0
        self.max_springs = structural + shear + bending

        self.positions = ti.Vector.field(3, ti.f32, shape=self.n_particles)
        self.velocities = ti.Vector.field(3, ti.f32, shape=self.n_particles)
        self.forces = ti.Vector.field(3, ti.f32, shape=self.n_particles)
        self.predicted_positions = ti.Vector.field(3, ti.f32, shape=self.n_particles)
        self.predicted_velocities = ti.Vector.field(3, ti.f32, shape=self.n_particles)
        self.predicted_forces = ti.Vector.field(3, ti.f32, shape=self.n_particles)
        self.fixed = ti.field(ti.i32, shape=self.n_particles)
        self.bind_positions = ti.Vector.field(3, ti.f32, shape=self.n_particles)

        self.spring_a = ti.field(ti.i32, shape=self.max_springs)
        self.spring_b = ti.field(ti.i32, shape=self.max_springs)
        self.spring_rest = ti.field(ti.f32, shape=self.max_springs)
        self.spring_type = ti.field(ti.i32, shape=self.max_springs)
        self.spring_count = ti.field(ti.i32, shape=())
        self.spring_type_counter = ti.field(ti.i32, shape=3)
        self.line_indices = ti.field(ti.i32, shape=max(1, self.max_springs * 2))

    @ti.func
    def particle_index(self, row, col):
        return row * self.cols + col

    @ti.func
    def add_spring(self, a, b, rest, spring_kind):
        spring_id = ti.atomic_add(self.spring_count[None], 1)
        self.spring_a[spring_id] = a
        self.spring_b[spring_id] = b
        self.spring_rest[spring_id] = rest
        self.spring_type[spring_id] = spring_kind
        ti.atomic_add(self.spring_type_counter[spring_kind], 1)

    @ti.kernel
    def initialize_positions_kernel(self):
        width = ti.cast(self.cols - 1, ti.f32) * self.spacing
        top_y = 0.72
        left_x = 0.5 - width * 0.5
        for particle_id in range(self.n_particles):
            row = particle_id // self.cols
            col = particle_id - row * self.cols
            position = ti.Vector(
                [
                    left_x + ti.cast(col, ti.f32) * self.spacing,
                    top_y - ti.cast(row, ti.f32) * self.spacing,
                    0.0,
                ]
            )
            self.positions[particle_id] = position
            self.bind_positions[particle_id] = position
            self.velocities[particle_id] = ti.Vector([0.0, 0.0, 0.0])
            self.forces[particle_id] = ti.Vector([0.0, 0.0, 0.0])
            self.predicted_positions[particle_id] = position
            self.predicted_velocities[particle_id] = ti.Vector([0.0, 0.0, 0.0])
            self.predicted_forces[particle_id] = ti.Vector([0.0, 0.0, 0.0])
            self.fixed[particle_id] = 1 if row == 0 else 0

    @ti.kernel
    def initialize_springs_kernel(self):
        self.spring_count[None] = 0
        for spring_kind in ti.static(range(3)):
            self.spring_type_counter[spring_kind] = 0

        for row, col in ti.ndrange(self.rows, self.cols):
            current = self.particle_index(row, col)
            if col + 1 < self.cols:
                self.add_spring(current, self.particle_index(row, col + 1), self.spacing, 0)
            if row + 1 < self.rows:
                self.add_spring(current, self.particle_index(row + 1, col), self.spacing, 0)
            if ti.static(self.enable_shear):
                if col + 1 < self.cols and row + 1 < self.rows:
                    self.add_spring(
                        current,
                        self.particle_index(row + 1, col + 1),
                        self.spacing * ti.sqrt(2.0),
                        1,
                    )
                    self.add_spring(
                        self.particle_index(row + 1, col),
                        self.particle_index(row, col + 1),
                        self.spacing * ti.sqrt(2.0),
                        1,
                    )
            if ti.static(self.enable_bending):
                if col + 2 < self.cols:
                    self.add_spring(current, self.particle_index(row, col + 2), self.spacing * 2.0, 2)
                if row + 2 < self.rows:
                    self.add_spring(current, self.particle_index(row + 2, col), self.spacing * 2.0, 2)

    @ti.kernel
    def initialize_render_indices_kernel(self):
        for spring_id in range(self.spring_count[None]):
            self.line_indices[spring_id * 2] = self.spring_a[spring_id]
            self.line_indices[spring_id * 2 + 1] = self.spring_b[spring_id]

    def initialize(self) -> None:
        self.initialize_positions_kernel()
        self.initialize_springs_kernel()
        self.initialize_render_indices_kernel()

    @ti.func
    def spring_stiffness_scale(self, spring_kind):
        scale = 1.0
        if spring_kind == 1:
            scale = 0.72
        if spring_kind == 2:
            scale = 0.32
        return scale

    @ti.func
    def atomic_add_force(self, particle_id, force):
        for k in ti.static(range(3)):
            ti.atomic_add(self.forces[particle_id][k], force[k])

    @ti.func
    def atomic_add_predicted_force(self, particle_id, force):
        for k in ti.static(range(3)):
            ti.atomic_add(self.predicted_forces[particle_id][k], force[k])

    @ti.func
    def compute_forces_on(self, spring_id, stiffness_scale):
        a = self.spring_a[spring_id]
        b = self.spring_b[spring_id]
        delta = self.positions[a] - self.positions[b]
        distance = delta.norm() + 1e-6
        direction = delta / distance
        stretch = distance - self.spring_rest[spring_id]
        spring_force = -self.stiffness * stiffness_scale * self.spring_stiffness_scale(self.spring_type[spring_id]) * stretch * direction
        relative_velocity = self.velocities[a] - self.velocities[b]
        damping_force = -self.damping * relative_velocity.dot(direction) * direction
        force = spring_force + damping_force
        self.atomic_add_force(a, force)
        self.atomic_add_force(b, -force)

    @ti.func
    def compute_predicted_forces_on(self, spring_id, stiffness_scale):
        a = self.spring_a[spring_id]
        b = self.spring_b[spring_id]
        delta = self.predicted_positions[a] - self.predicted_positions[b]
        distance = delta.norm() + 1e-6
        direction = delta / distance
        stretch = distance - self.spring_rest[spring_id]
        spring_force = -self.stiffness * stiffness_scale * self.spring_stiffness_scale(self.spring_type[spring_id]) * stretch * direction
        relative_velocity = self.predicted_velocities[a] - self.predicted_velocities[b]
        damping_force = -self.damping * relative_velocity.dot(direction) * direction
        force = spring_force + damping_force
        self.atomic_add_predicted_force(a, force)
        self.atomic_add_predicted_force(b, -force)

    @ti.func
    def clamp_velocity(self, velocity, max_speed):
        speed = velocity.norm()
        result = velocity
        if speed > max_speed:
            result = velocity / speed * max_speed
        return result

    @ti.func
    def restore_fixed_particle(self, particle_id):
        self.positions[particle_id] = self.bind_positions[particle_id]
        self.velocities[particle_id] = ti.Vector([0.0, 0.0, 0.0])

    @ti.func
    def project_sphere_collision(self, particle_id):
        center = ti.Vector([self.sphere_center_np[0], self.sphere_center_np[1], self.sphere_center_np[2]])
        offset = self.positions[particle_id] - center
        distance = offset.norm()
        if distance < self.sphere_radius:
            normal = ti.Vector([0.0, 1.0, 0.0])
            if distance > 1e-6:
                normal = offset / distance
            self.positions[particle_id] = center + normal * self.sphere_radius
            normal_speed = self.velocities[particle_id].dot(normal)
            if normal_speed < 0.0:
                self.velocities[particle_id] -= normal_speed * normal

    @ti.func
    def clear_force_for_particle(self, particle_id, gravity_y):
        gravity_force = ti.Vector([0.0, -gravity_y * self.mass, 0.0])
        damping_force = -self.damping * self.velocities[particle_id]
        self.forces[particle_id] = gravity_force + damping_force

    @ti.kernel
    def step_explicit(self, dt: ti.f32, gravity_y: ti.f32, stiffness_scale: ti.f32, max_speed: ti.f32):
        for particle_id in range(self.n_particles):
            self.clear_force_for_particle(particle_id, gravity_y)

        for spring_id in range(self.spring_count[None]):
            self.compute_forces_on(spring_id, stiffness_scale)

        for particle_id in range(self.n_particles):
            if self.fixed[particle_id] == 1:
                self.restore_fixed_particle(particle_id)
            else:
                acceleration = self.forces[particle_id] / self.mass
                old_velocity = self.velocities[particle_id]
                self.positions[particle_id] += old_velocity * dt
                self.velocities[particle_id] = self.clamp_velocity(old_velocity + acceleration * dt, max_speed)
                self.project_sphere_collision(particle_id)

    @ti.kernel
    def step_semi_implicit(self, dt: ti.f32, gravity_y: ti.f32, stiffness_scale: ti.f32, max_speed: ti.f32):
        for particle_id in range(self.n_particles):
            self.clear_force_for_particle(particle_id, gravity_y)

        for spring_id in range(self.spring_count[None]):
            self.compute_forces_on(spring_id, stiffness_scale)

        for particle_id in range(self.n_particles):
            if self.fixed[particle_id] == 1:
                self.restore_fixed_particle(particle_id)
            else:
                acceleration = self.forces[particle_id] / self.mass
                self.velocities[particle_id] = self.clamp_velocity(self.velocities[particle_id] + acceleration * dt, max_speed)
                self.positions[particle_id] += self.velocities[particle_id] * dt
                self.project_sphere_collision(particle_id)

    @ti.kernel
    def step_implicit_iter(
        self,
        dt: ti.f32,
        gravity_y: ti.f32,
        stiffness_scale: ti.f32,
        max_speed: ti.f32,
        iterations: ti.i32,
    ):
        for particle_id in range(self.n_particles):
            self.predicted_positions[particle_id] = self.positions[particle_id] + self.velocities[particle_id] * dt
            self.predicted_velocities[particle_id] = self.velocities[particle_id]

        for iteration in range(8):
            if iteration < iterations:
                for particle_id in range(self.n_particles):
                    gravity_force = ti.Vector([0.0, -gravity_y * self.mass, 0.0])
                    damping_force = -self.damping * self.predicted_velocities[particle_id]
                    self.predicted_forces[particle_id] = gravity_force + damping_force

                for spring_id in range(self.spring_count[None]):
                    self.compute_predicted_forces_on(spring_id, stiffness_scale)

                for particle_id in range(self.n_particles):
                    if self.fixed[particle_id] == 0:
                        acceleration = self.predicted_forces[particle_id] / self.mass
                        self.predicted_velocities[particle_id] = self.clamp_velocity(
                            self.velocities[particle_id] + acceleration * dt,
                            max_speed,
                        )
                        self.predicted_positions[particle_id] = self.positions[particle_id] + self.predicted_velocities[particle_id] * dt

        for particle_id in range(self.n_particles):
            if self.fixed[particle_id] == 1:
                self.restore_fixed_particle(particle_id)
            else:
                self.velocities[particle_id] = self.predicted_velocities[particle_id]
                self.positions[particle_id] = self.predicted_positions[particle_id]
                self.project_sphere_collision(particle_id)

    @ti.kernel
    def project_collisions_kernel(self):
        for particle_id in range(self.n_particles):
            if self.fixed[particle_id] == 0:
                self.project_sphere_collision(particle_id)

    @ti.kernel
    def set_particle_position_kernel(self, particle_id: ti.i32, x: ti.f32, y: ti.f32, z: ti.f32):
        self.positions[particle_id] = ti.Vector([x, y, z])
        self.velocities[particle_id] = ti.Vector([0.0, 0.0, 0.0])

    def step(self, method: Integrator, dt: float = 1.0 / 120.0, gravity_y: float = 9.8, stiffness_scale: float = 1.0) -> None:
        if method == "explicit":
            self.step_explicit(dt, gravity_y, stiffness_scale, self.max_speed)
        elif method == "semi_implicit":
            self.step_semi_implicit(dt, gravity_y, stiffness_scale, self.max_speed)
        elif method == "implicit":
            self.step_implicit_iter(dt, gravity_y, stiffness_scale, self.max_speed, 5)
        else:
            raise ValueError(f"unknown integrator: {method}")

    def spring_type_counts(self) -> dict[str, int]:
        counts = self.spring_type_counter.to_numpy()
        return {
            "structural": int(counts[0]),
            "shear": int(counts[1]),
            "bending": int(counts[2]),
        }

    def positions_numpy(self) -> np.ndarray:
        return self.positions.to_numpy()

    def line_indices_numpy(self) -> np.ndarray:
        return self.line_indices.to_numpy()[: int(self.spring_count[None]) * 2]

    def set_particle_position(self, particle_id: int, position: list[float] | tuple[float, float, float]) -> None:
        self.set_particle_position_kernel(particle_id, float(position[0]), float(position[1]), float(position[2]))

    def project_collisions_only(self) -> None:
        self.project_collisions_kernel()


def save_numpy_snapshot(cloth: MassSpringCloth, output_path: str | Path) -> None:
    """Save a lightweight text snapshot for debugging and grading logs."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, cloth.positions_numpy(), fmt="%.6f")
