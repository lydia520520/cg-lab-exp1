"""
实验五选做扩展版本
选做 1：折射与玻璃材质（+15%）
  - 红球改为玻璃材质，折射率 IOR=1.5
  - 使用斯涅尔定律计算透射方向
  - 发生全反射时转为镜面反射处理
选做 2：抗锯齿 MSAA（+10%）
  - 每像素发射多条主射线（可通过滑条调节）
  - 对结果取平均，消除边缘锯齿
"""
import taichi as ti

ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# 交互参数
light_pos_x  = ti.field(ti.f32, shape=())
light_pos_y  = ti.field(ti.f32, shape=())
light_pos_z  = ti.field(ti.f32, shape=())
max_bounces  = ti.field(ti.i32, shape=())
aa_samples   = ti.field(ti.i32, shape=())  # 每像素采样数（抗锯齿）

# 材质枚举
MAT_DIFFUSE = 0
MAT_MIRROR  = 1
MAT_GLASS   = 2   # 选做 1：玻璃折射材质

GLASS_IOR = 1.5   # 玻璃折射率

@ti.func
def normalize(v):
    return v / v.norm(1e-5)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

@ti.func
def refract(I, N, eta):
    """
    斯涅尔定律折射，返回 (是否发生全反射, 折射方向)
    eta = n1 / n2
    I 为归一化入射方向，N 为法线（朝向入射侧）
    """
    cos_i = -I.dot(N)
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)
    total_internal = sin2_t > 1.0
    refracted = ti.Vector([0.0, 0.0, 0.0])
    if not total_internal:
        cos_t = ti.sqrt(1.0 - sin2_t)
        refracted = normalize(eta * I + (eta * cos_i - cos_t) * N)
    return total_internal, refracted

@ti.func
def intersect_sphere(ro, rd, center, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        t2 = (-b + ti.sqrt(delta)) / 2.0
        if t1 > 1e-4:
            t = t1
        elif t2 > 1e-4:
            t = t2   # 射线从玻璃球内部出射时取 t2
        if t > 0:
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal

@ti.func
def intersect_plane(ro, rd, plane_y):
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(rd.y) > 1e-5:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 1e-4:
            t = t1
    return t, normal

@ti.func
def scene_intersect(ro, rd):
    min_t  = 1e10
    hit_n  = ti.Vector([0.0, 0.0, 0.0])
    hit_c  = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    # 1. 玻璃球（原红球位置，选做 1）
    t, n = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.95, 0.95, 1.0])  # 淡蓝色玻璃基础色
        hit_mat = MAT_GLASS

    # 2. 银色镜面球
    t, n = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.9, 0.9, 0.9])
        hit_mat = MAT_MIRROR

    # 3. 地板（棋盘格纹理）
    t, n = intersect_plane(ro, rd, -1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_mat = MAT_DIFFUSE
        p = ro + rd * t
        ix = ti.floor(p.x * 2.0)
        iz = ti.floor(p.z * 2.0)
        if (ix + iz) % 2 == 0:
            hit_c = ti.Vector([0.3, 0.3, 0.3])
        else:
            hit_c = ti.Vector([0.8, 0.8, 0.8])

    return min_t, hit_n, hit_c, hit_mat

@ti.func
def trace_ray(ro, rd, light_pos, bg_color, n_bounces):
    """追踪单条光线，返回颜色"""
    final_color = ti.Vector([0.0, 0.0, 0.0])
    throughput  = ti.Vector([1.0, 1.0, 1.0])

    for bounce in range(n_bounces):
        t, N, obj_color, mat_id = scene_intersect(ro, rd)

        if t > 1e9:
            final_color += throughput * bg_color
            break

        p = ro + rd * t

        if mat_id == MAT_MIRROR:
            ro = p + N * 1e-4
            rd = normalize(reflect(rd, N))
            throughput *= 0.8 * obj_color

        elif mat_id == MAT_GLASS:
            # 判断射线从外部还是内部入射（法线与入射方向点积符号）
            entering = rd.dot(N) < 0.0
            eta = 1.0 / GLASS_IOR if entering else GLASS_IOR / 1.0
            # 保证折射计算时法线朝向入射侧
            fwd_N = N if entering else -N

            total_internal, refracted = refract(rd, fwd_N, eta)

            if total_internal:
                # 全内反射：按镜面处理
                ro = p + fwd_N * 1e-4
                rd = normalize(reflect(rd, fwd_N))
            else:
                # 折射：沿法线反向偏移，穿透球体
                ro = p - fwd_N * 1e-4
                rd = refracted

            throughput *= obj_color  # 玻璃轻微吸收

        elif mat_id == MAT_DIFFUSE:
            L = normalize(light_pos - p)
            shadow_orig = p + N * 1e-4
            shadow_t, _, _, _ = scene_intersect(shadow_orig, L)
            dist_to_light = (light_pos - p).norm()

            in_shadow = 0.0
            if 0 < shadow_t < dist_to_light:
                in_shadow = 1.0

            ambient = 0.2 * obj_color
            direct_light = ambient
            if in_shadow == 0.0:
                diff = ti.max(0.0, N.dot(L))
                direct_light += 0.8 * diff * obj_color

            final_color += throughput * direct_light
            break

    return final_color

@ti.kernel
def render():
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    bg_color  = ti.Vector([0.05, 0.15, 0.2])
    n_bounces = max_bounces[None]
    n_samples = aa_samples[None]

    for i, j in pixels:
        color_acc = ti.Vector([0.0, 0.0, 0.0])

        # 选做 2：MSAA 多重采样抗锯齿
        for s in range(n_samples):
            # 在像素内随机偏移（[0,1) 均匀采样）
            dx = ti.random(ti.f32) if n_samples > 1 else 0.5
            dy = ti.random(ti.f32) if n_samples > 1 else 0.5

            u = (i + dx - res_x / 2.0) / res_y * 2.0
            v = (j + dy - res_y / 2.0) / res_y * 2.0

            ro = ti.Vector([0.0, 1.0, 5.0])
            rd = normalize(ti.Vector([u, v - 0.2, -1.0]))

            color_acc += trace_ray(ro, rd, light_pos, bg_color, n_bounces)

        pixels[i, j] = ti.math.clamp(color_acc / n_samples, 0.0, 1.0)

def main():
    window = ti.ui.Window("Ray Tracing Extra (Glass + MSAA)", (res_x, res_y))
    canvas = window.get_canvas()
    gui    = window.get_gui()

    light_pos_x[None] = 2.0
    light_pos_y[None] = 4.0
    light_pos_z[None] = 3.0
    max_bounces[None]  = 5
    aa_samples[None]   = 1

    while window.running:
        render()
        canvas.set_image(pixels)

        with gui.sub_window("Controls", 0.72, 0.05, 0.26, 0.28):
            light_pos_x[None] = gui.slider_float('Light X',      light_pos_x[None], -5.0, 5.0)
            light_pos_y[None] = gui.slider_float('Light Y',      light_pos_y[None],  1.0, 8.0)
            light_pos_z[None] = gui.slider_float('Light Z',      light_pos_z[None], -5.0, 5.0)
            max_bounces[None]  = gui.slider_int('Max Bounces',   max_bounces[None],  1, 8)
            aa_samples[None]   = gui.slider_int('AA Samples',    aa_samples[None],   1, 8)

        window.show()

if __name__ == '__main__':
    main()
