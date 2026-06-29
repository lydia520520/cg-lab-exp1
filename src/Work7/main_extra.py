from __future__ import annotations

import argparse

import taichi as ti
from taichi_mass_spring import MassSpringCloth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Work7 optional part: one-click Taichi visualization window.")
    parser.add_argument("--max-frames", type=int, default=0, help="Auto-exit after N frames. 0 means run until the window is closed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cloth = MassSpringCloth(
        cols=24,
        rows=20,
        spacing=0.034,
        stiffness=700.0,
        damping=3.6,
        enable_shear=True,
        enable_bending=True,
        sphere_center=(0.5, 0.12, 0.0),
        sphere_radius=0.18,
    )
    cloth.initialize()

    sphere_center_field = ti.Vector.field(3, dtype=ti.f32, shape=1)
    sphere_center_field[0] = ti.Vector(list(cloth.sphere_center_np))

    window = ti.ui.Window("Work7 Optional Demo - Cloth With Shear, Bending, and Sphere Collision", (1320, 860), vsync=True)
    canvas = window.get_canvas()
    gui = window.get_gui()
    camera = ti.ui.Camera()
    camera.position(0.46, 0.40, 1.72)
    camera.lookat(0.50, 0.16, 0.00)
    camera.up(0.0, 1.0, 0.0)

    integrators = ["explicit", "semi_implicit", "implicit"]
    integrator_index = 1
    paused = False
    gravity = 9.8
    stiffness_scale = 1.0
    frame_count = 0

    while window.running:
        if window.is_pressed("r"):
            cloth.initialize()

        with gui.sub_window("Optional Controls", 0.02, 0.02, 0.29, 0.32):
            gui.text("Enabled optional features")
            gui.text("- shear springs")
            gui.text("- bending springs")
            gui.text("- sphere collision")
            if gui.button("Explicit Euler"):
                integrator_index = 0
            if gui.button("Semi-Implicit Euler"):
                integrator_index = 1
            if gui.button("Implicit Euler"):
                integrator_index = 2
            if gui.button("Pause / Resume"):
                paused = not paused
            if gui.button("Reset"):
                cloth.initialize()
            gravity = gui.slider_float("gravity", gravity, 0.0, 18.0)
            stiffness_scale = gui.slider_float("stiffness", stiffness_scale, 0.3, 1.8)
            gui.text(f"Current: {integrators[integrator_index]}")
            gui.text("R: reset cloth")

        if not paused:
            cloth.step(integrators[integrator_index], dt=1.0 / 120.0, gravity_y=gravity, stiffness_scale=stiffness_scale)

        scene = window.get_scene()
        canvas.set_background_color((0.95, 0.97, 1.0))
        scene.set_camera(camera)
        scene.ambient_light((0.72, 0.74, 0.78))
        scene.point_light(pos=(0.18, 1.05, 1.20), color=(1.0, 0.98, 0.96))
        scene.point_light(pos=(0.92, 0.68, 0.46), color=(0.70, 0.78, 1.0))
        scene.particles(sphere_center_field, radius=cloth.sphere_radius, color=(0.20, 0.45, 0.95))
        scene.lines(
            cloth.positions,
            indices=cloth.line_indices,
            width=1.7,
            color=(0.10, 0.63, 0.56),
        )
        scene.particles(cloth.positions, radius=0.0048, color=(0.08, 0.12, 0.22))
        canvas.scene(scene)
        window.show()

        frame_count += 1
        if args.max_frames > 0 and frame_count >= args.max_frames:
            break


if __name__ == "__main__":
    main()
