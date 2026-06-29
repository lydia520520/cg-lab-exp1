from __future__ import annotations

import argparse

import taichi as ti
from taichi_mass_spring import MassSpringCloth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Work7 required part: one-click Taichi visualization window.")
    parser.add_argument("--max-frames", type=int, default=0, help="Auto-exit after N frames. 0 means run until the window is closed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cloth = MassSpringCloth(
        cols=20,
        rows=20,
        spacing=0.035,
        enable_shear=False,
        enable_bending=False,
        sphere_radius=0.0,
    )
    cloth.initialize()

    window = ti.ui.Window("Work7 Mass-Spring Model - Xu Yijia", (1024, 768), vsync=True)
    canvas = window.get_canvas()
    gui = window.get_gui()
    camera = ti.ui.Camera()
    camera.position(0.5, 0.25, 1.8)
    camera.lookat(0.5, 0.25, 0.0)
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

        with gui.sub_window("Controls", 0.02, 0.02, 0.28, 0.26):
            gui.text("Integrator")
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
        scene.set_camera(camera)
        scene.ambient_light((0.55, 0.55, 0.55))
        scene.point_light(pos=(0.2, 0.9, 1.0), color=(1.0, 1.0, 1.0))
        scene.lines(
            cloth.positions,
            indices=cloth.line_indices,
            width=1.5,
            color=(0.18, 0.42, 0.95),
        )
        scene.particles(cloth.positions, radius=0.004, color=(0.08, 0.12, 0.16))
        canvas.scene(scene)
        window.show()
        frame_count += 1
        if args.max_frames > 0 and frame_count >= args.max_frames:
            break


if __name__ == "__main__":
    main()
