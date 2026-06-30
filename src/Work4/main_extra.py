from __future__ import annotations

import argparse

import taichi as ti


RES_X, RES_Y = 960, 720

pixels = None

ka = None
kd = None
ks = None
shininess = None

use_blinn = None
use_hard_shadow = None
light_angle = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Work4 optional: Blinn-Phong and hard shadow demo.")
    parser.add_argument("--arch", choices=["cpu", "gpu"], default="cpu", help="Taichi backend.")
    parser.add_argument("--headless", action="store_true", help="Run without opening the GUI window.")
    parser.add_argument("--max-frames", type=int, default=0, help="Exit after N frames for automated checks. 0 means no limit.")
    return parser.parse_args()


def init_taichi(arch: str) -> None:
    global pixels, ka, kd, ks, shininess, use_blinn, use_hard_shadow, light_angle
    ti.init(arch=ti.cpu if arch == "cpu" else ti.gpu)
    pixels = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
    ka = ti.field(dtype=ti.f32, shape=())
    kd = ti.field(dtype=ti.f32, shape=())
    ks = ti.field(dtype=ti.f32, shape=())
    shininess = ti.field(dtype=ti.f32, shape=())
    use_blinn = ti.field(dtype=ti.i32, shape=())
    use_hard_shadow = ti.field(dtype=ti.i32, shape=())
    light_angle = ti.field(dtype=ti.f32, shape=())


@ti.func
def normalize(v):
    return v / max(v.norm(), 1e-5)


@ti.func
def reflect(i, n):
    return i - 2.0 * i.dot(n) * n


@ti.func
def intersect_sphere(ro, rd, center, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0.0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        if t1 > 1e-4:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal


@ti.func
def intersect_cone(ro, rd, apex, base_y, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    h = apex.y - base_y
    k = (radius / h) ** 2
    ro_local = ro - apex

    a = rd.x * rd.x + rd.z * rd.z - k * rd.y * rd.y
    b = 2.0 * (ro_local.x * rd.x + ro_local.z * rd.z - k * ro_local.y * rd.y)
    c = ro_local.x * ro_local.x + ro_local.z * ro_local.z - k * ro_local.y * ro_local.y

    if ti.abs(a) > 1e-5:
        delta = b * b - 4.0 * a * c
        if delta > 0.0:
            t1 = (-b - ti.sqrt(delta)) / (2.0 * a)
            t2 = (-b + ti.sqrt(delta)) / (2.0 * a)
            t_first = t1
            t_second = t2
            if t1 > t2:
                t_first, t_second = t_second, t_first

            y1 = ro_local.y + t_first * rd.y
            if t_first > 1e-4 and -h <= y1 <= 0.0:
                t = t_first
            else:
                y2 = ro_local.y + t_second * rd.y
                if t_second > 1e-4 and -h <= y2 <= 0.0:
                    t = t_second

            if t > 0.0:
                p_local = ro_local + rd * t
                normal = normalize(ti.Vector([p_local.x, -k * p_local.y, p_local.z]))

    return t, normal


@ti.func
def intersect_plane(ro, rd, plane_y):
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(rd.y) > 1e-5:
        candidate = (plane_y - ro.y) / rd.y
        if candidate > 1e-4:
            t = candidate
    return t, normal


@ti.func
def intersect_scene(ro, rd):
    hit = 0
    min_t = 1e10
    normal = ti.Vector([0.0, 0.0, 0.0])
    color = ti.Vector([0.0, 0.0, 0.0])

    t_sphere, n_sphere = intersect_sphere(ro, rd, ti.Vector([-1.25, -0.25, -0.2]), 1.2)
    if 0.0 < t_sphere < min_t:
        hit = 1
        min_t = t_sphere
        normal = n_sphere
        color = ti.Vector([0.85, 0.18, 0.18])

    t_cone, n_cone = intersect_cone(ro, rd, ti.Vector([1.35, 1.25, -0.2]), -1.55, 1.2)
    if 0.0 < t_cone < min_t:
        hit = 1
        min_t = t_cone
        normal = n_cone
        color = ti.Vector([0.58, 0.30, 0.88])

    t_plane, n_plane = intersect_plane(ro, rd, -1.55)
    if 0.0 < t_plane < min_t:
        hit = 1
        min_t = t_plane
        normal = n_plane
        checker = (ti.floor((ro.x + rd.x * t_plane) * 1.2) + ti.floor((ro.z + rd.z * t_plane) * 1.2)) % 2
        base = 0.72 + 0.12 * checker
        color = ti.Vector([base, base, base])

    return hit, min_t, normal, color


@ti.func
def in_shadow(point, light_pos):
    shadow_origin = point + ti.Vector([0.0, 0.0, 0.0])
    shadow_dir = normalize(light_pos - point)
    shadow_origin += shadow_dir * 1e-3
    distance_to_light = (light_pos - point).norm()
    hit, t_shadow, _, _ = intersect_scene(shadow_origin, shadow_dir)
    return hit == 1 and t_shadow > 0.0 and t_shadow < distance_to_light


@ti.kernel
def render():
    light_pos = ti.Vector([3.4 * ti.cos(light_angle[None]), 3.5, 2.6 + 2.6 * ti.sin(light_angle[None])])
    eye = ti.Vector([0.0, 0.35, 5.6])

    for i, j in pixels:
        u = (i - RES_X * 0.5) / RES_Y * 2.0
        v = (j - RES_Y * 0.5) / RES_Y * 2.0
        ro = eye
        rd = normalize(ti.Vector([u, v, -1.35]))

        hit, min_t, hit_normal, hit_color = intersect_scene(ro, rd)
        color = ti.Vector([0.07, 0.10, 0.14])

        if hit == 1:
            point = ro + rd * min_t
            n = hit_normal
            l = normalize(light_pos - point)
            v_dir = normalize(ro - point)

            ambient = ka[None] * hit_color
            diffuse = ti.Vector([0.0, 0.0, 0.0])
            specular = ti.Vector([0.0, 0.0, 0.0])

            shadowed = 0
            if use_hard_shadow[None] == 1 and in_shadow(point, light_pos):
                shadowed = 1

            if shadowed == 0:
                diff = ti.max(0.0, n.dot(l))
                diffuse = kd[None] * diff * hit_color

                spec = 0.0
                if use_blinn[None] == 1:
                    h = normalize(l + v_dir)
                    spec = ti.pow(ti.max(0.0, n.dot(h)), shininess[None])
                else:
                    r = normalize(reflect(-l, n))
                    spec = ti.pow(ti.max(0.0, r.dot(v_dir)), shininess[None])
                specular = ks[None] * spec * ti.Vector([1.0, 1.0, 1.0])

            color = ambient + diffuse + specular

        pixels[i, j] = ti.math.clamp(color, 0.0, 1.0)


def main() -> None:
    args = parse_args()
    init_taichi(args.arch)

    ka[None] = 0.18
    kd[None] = 0.76
    ks[None] = 0.56
    shininess[None] = 48.0
    use_blinn[None] = 1
    use_hard_shadow[None] = 1
    light_angle[None] = 0.0

    if args.headless:
        frame = 0
        while True:
            render()
            light_angle[None] += 0.03
            frame += 1
            if args.max_frames and frame >= args.max_frames:
                break
        return

    window = ti.ui.Window("Work4 Extra - Blinn-Phong and Hard Shadow", (RES_X, RES_Y))
    canvas = window.get_canvas()
    gui = window.get_gui()

    frame = 0
    auto_rotate_light = True

    while window.running:
        if args.max_frames and frame >= args.max_frames:
            break

        for event in window.get_events(ti.ui.PRESS):
            if event.key == ti.ui.ESCAPE:
                window.running = False
            elif event.key in ("b", "B"):
                use_blinn[None] = 1 - use_blinn[None]
            elif event.key in ("h", "H"):
                use_hard_shadow[None] = 1 - use_hard_shadow[None]
            elif event.key in ("l", "L"):
                auto_rotate_light = not auto_rotate_light

        if auto_rotate_light:
            light_angle[None] += 0.02

        render()
        canvas.set_image(pixels)

        with gui.sub_window("Optional Controls", 0.69, 0.05, 0.28, 0.28):
            ka[None] = gui.slider_float("Ka", ka[None], 0.0, 0.6)
            kd[None] = gui.slider_float("Kd", kd[None], 0.0, 1.2)
            ks[None] = gui.slider_float("Ks", ks[None], 0.0, 1.2)
            shininess[None] = gui.slider_float("Shininess", shininess[None], 4.0, 128.0)
            gui.text(f"Specular: {'Blinn-Phong' if use_blinn[None] == 1 else 'Phong'}")
            gui.text(f"Hard shadow: {'On' if use_hard_shadow[None] == 1 else 'Off'}")
            gui.text(f"Light rotate: {'On' if auto_rotate_light else 'Off'}")
            gui.text("B: switch specular model")
            gui.text("H: toggle hard shadow")
            gui.text("L: toggle moving light")

        window.show()
        frame += 1


if __name__ == "__main__":
    main()
