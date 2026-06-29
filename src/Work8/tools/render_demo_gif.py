from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "demo.gif"
WIDTH, HEIGHT = 640, 420
LENGTHS = [95, 82, 58]
JOINT_X = [0, LENGTHS[0], LENGTHS[0] + LENGTHS[1], sum(LENGTHS)]
PALETTE = [(45, 109, 246), (0, 153, 136), (233, 116, 36), (185, 88, 216)]


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def matmul(a, b):
    return [
        a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
        a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
        a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
        a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
        a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
        a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
        0,
        0,
        1,
    ]


def translate(x, y):
    return [1, 0, x, 0, 1, y, 0, 0, 1]


def rotate(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return [c, -s, 0, s, c, 0, 0, 0, 1]


def transform(m, p):
    return (m[0] * p[0] + m[1] * p[1] + m[2], m[3] * p[0] + m[4] * p[1] + m[5])


def invert_rigid(m):
    r00, r01, r10, r11, tx, ty = m[0], m[1], m[3], m[4], m[2], m[5]
    return [r00, r10, -(r00 * tx + r10 * ty), r01, r11, -(r01 * tx + r11 * ty), 0, 0, 1]


def pose(angles):
    offsets = [(0, 0), (LENGTHS[0], 0), (LENGTHS[1], 0), (LENGTHS[2], 0)]
    keys = ["shoulder", "elbow", "wrist", "tip"]
    global_mats = []
    for i, (x, y) in enumerate(offsets):
        local = translate(x, y)
        if keys[i] != "tip":
            local = matmul(local, rotate(angles.get(keys[i], 0)))
        global_mats.append(local if i == 0 else matmul(global_mats[i - 1], local))
    return global_mats


BIND = pose({"shoulder": 0, "elbow": 0, "wrist": 0})
INV_BIND = [invert_rigid(m) for m in BIND]


def mesh(segments=30, half_width=20):
    vertices = []
    faces = []
    for i in range(segments + 1):
        x = JOINT_X[-1] * i / segments
        taper = 0.58 + 0.42 * (1 - i / segments)
        vertices.append((x, -half_width * taper))
        vertices.append((x, half_width * taper))
    for i in range(segments):
        a = i * 2
        faces.extend([(a, a + 1, a + 3), (a, a + 3, a + 2)])
    return vertices, faces


VERTICES, FACES = mesh()


def weights_for_vertex(vertex, radius):
    x = vertex[0]
    if x <= radius:
        return [(0, 1.0)]
    if x >= JOINT_X[-1] - radius:
        return [(3, 1.0)]
    left = max(i for i in range(len(JOINT_X) - 1) if JOINT_X[i] <= x)
    right = min(left + 1, 3)
    if abs(x - JOINT_X[right]) < radius:
        t = max(0, min(1, (x - (JOINT_X[right] - radius)) / (2 * radius)))
        return [(left, 1 - t), (right, t)]
    if left > 0 and abs(x - JOINT_X[left]) < radius:
        t = max(0, min(1, (x - (JOINT_X[left] - radius)) / (2 * radius)))
        return [(left - 1, 1 - t), (left, t)]
    return [(left, 1.0)]


def blended_color(influences):
    rgb = [0, 0, 0]
    for joint, weight in influences:
        color = PALETTE[joint]
        rgb[0] += color[0] * weight
        rgb[1] += color[1] * weight
        rgb[2] += color[2] * weight
    return tuple(round(c) for c in rgb)


def skin(vertices, influences, mats):
    skin_mats = [matmul(mats[i], INV_BIND[i]) for i in range(4)]
    out = []
    for vertex, influence in zip(vertices, influences):
        x, y = 0, 0
        for joint, weight in influence:
            sx, sy = transform(skin_mats[joint], vertex)
            x += sx * weight
            y += sy * weight
        out.append((x, y))
    return out


def stage(p):
    scale = min(WIDTH / 390, HEIGHT / 285)
    return (WIDTH * 0.19 + p[0] * scale, HEIGHT * 0.57 - p[1] * scale)


def dashed(draw, a, b, fill, width=1, step=10):
    ax, ay = a
    bx, by = b
    dist = math.hypot(bx - ax, by - ay)
    pieces = max(1, int(dist / step))
    for i in range(0, pieces, 2):
        t0 = i / pieces
        t1 = min(1, (i + 1) / pieces)
        draw.line((ax + (bx - ax) * t0, ay + (by - ay) * t0, ax + (bx - ax) * t1, ay + (by - ay) * t1), fill=fill, width=width)


def render(frame):
    radius = 18 + 16 * (0.5 + 0.5 * math.sin(frame * 0.12))
    influences = [weights_for_vertex(vertex, radius) for vertex in VERTICES]
    angles = {
        "shoulder": math.radians(math.sin(frame * 0.08) * 34 - 8),
        "elbow": math.radians(math.sin(frame * 0.11 + 0.8) * 68),
        "wrist": math.radians(math.sin(frame * 0.14 + 1.9) * 54),
    }
    mats = pose(angles)
    skinned = skin(VERTICES, influences, mats)

    img = Image.new("RGB", (WIDTH, HEIGHT), "#fbfdfc")
    draw = ImageDraw.Draw(img, "RGBA")
    for x in range(0, WIDTH, 40):
        draw.line((x, 0, x, HEIGHT), fill="#dfe8df")
    for y in range(0, HEIGHT, 40):
        draw.line((0, y, WIDTH, y), fill="#dfe8df")

    for face in FACES:
        pts = [stage(skinned[i]) for i in face]
        color = blended_color(influences[face[0]]) + (142,)
        draw.polygon(pts, fill=color)
        draw.line((*pts[0], *pts[1]), fill=(24, 34, 48, 36), width=1)
        draw.line((*pts[1], *pts[2]), fill=(24, 34, 48, 36), width=1)
        draw.line((*pts[2], *pts[0]), fill=(24, 34, 48, 36), width=1)

    bind_points = [stage((x, 0)) for x in JOINT_X]
    for a, b in zip(bind_points, bind_points[1:]):
        dashed(draw, a, b, (185, 88, 216, 138), 3)

    joint_points = [stage((m[2], m[5])) for m in mats]
    for i, (a, b) in enumerate(zip(joint_points, joint_points[1:])):
        draw.line((*a, *b), fill=PALETTE[i] + (255,), width=6)
    for p in joint_points:
        draw.ellipse((p[0] - 7, p[1] - 7, p[0] + 7, p[1] + 7), fill=(255, 255, 255, 255), outline=(24, 34, 48, 255), width=2)

    title = font(16)
    small = font(12)
    draw.text((20, HEIGHT - 48), "Work8 Linear Blend Skinning", fill=(24, 34, 48, 255), font=title)
    draw.text((20, HEIGHT - 26), "joint animation + weights + bind-pose comparison | Author: Xu Yijia", fill=(92, 102, 117, 255), font=small)
    return img


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames = [render(frame) for frame in range(76)]
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=45, loop=0, optimize=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
