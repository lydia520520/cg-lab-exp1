202311051007-许艺珈-人工智能
# 计算机图形学 实验五：光线追踪

## 效果演示（必做+选做）

![必做](https://github.com/lydia520520/cg-lab/blob/main/assets/Work51.gif)
![选做](https://github.com/lydia520520/cg-lab/blob/main/assets/Work52.gif)
(如长时间未加载出gif 可查看assets/Work51.gif)
(如长时间未加载出gif 可查看assets/Work52.gif)

## 实验目标

- 理解光线投射（Ray Casting）与光线追踪（Ray Tracing）的本质区别
- 掌握 Whitted-Style 光线追踪中硬阴影与理想镜面反射的实现方法
- 学习如何将递归光线追踪改写为适合 GPU 并行计算的迭代（循环）模式

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `ray_tracing.py` | 必做部分：完整的 Whitted-Style 光线追踪实现 |
| `ray_tracing_extra.py` | 选做部分：折射玻璃材质 + MSAA 抗锯齿 |

---

## 环境配置

使用 uv 进行环境管理：

```bash
# 创建虚拟环境（Python 3.12）
uv venv --python 3.12

# 激活环境
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

# 安装 Taichi
uv pip install taichi
```

---

## 运行方式

```bash
source .venv/bin/activate   # macOS / Linux

# 必做版本
python ray_tracing.py

# 选做扩展版本（玻璃折射 + 抗锯齿）
python ray_tracing_extra.py
```

---

## 必做任务说明

### 任务 1：三维场景搭建

场景包含三个几何体，均在 Taichi Kernel 中隐式定义：

- **无限大地板**：位于 `y = -1.0`，法线朝上，带黑白棋盘格纹理（通过交点 x、z 坐标奇偶性判断颜色），漫反射材质
- **红色漫反射球**：圆心 `(-1.2, 0, 0)`，半径 1.0，漫反射材质
- **银色镜面球**：圆心 `(1.2, 0, 0)`，半径 1.0，纯镜面反射材质

### 任务 2：迭代式光线弹射

以 `for` 循环（最大 `max_bounces` 次）替代递归，追踪光线路径：

- `throughput`（吞吐量）初始为 `(1, 1, 1)`，每次击中镜面时乘以反射率衰减
- 击中镜面：更新光线起点与方向，继续下一次循环
- 击中漫反射：计算光照颜色乘以 `throughput`，累加到 `final_color`，然后 `break`

### 任务 3：硬阴影 + Shadow Acne 修复

- 从漫反射交点向光源发射暗影射线，若被遮挡则该点仅保留环境光
- **关键**：反射射线和暗影射线的起点需沿法线方向偏移 `1e-4`，防止自相交（Shadow Acne）产生黑色噪点

### 任务 4：UI 交互面板

使用 `ti.ui.Window` 提供以下滑动条，运行时实时更新：

| 控件 | 范围 | 说明 |
|------|------|------|
| Light X / Y / Z | 各轴范围 | 实时移动点光源，观察阴影变化 |
| Max Bounces | 1 ~ 5 | 调整最大弹射次数，观察镜面反射层数 |

---

## 选做任务说明（`ray_tracing_extra.py`）

### 选做 1：折射与玻璃材质（+15%）

将左侧红球替换为玻璃球（折射率 IOR = 1.5），实现斯涅尔定律折射：

$$\mathbf{T} = \eta \, \mathbf{I} + \left(\eta \cos\theta_i - \cos\theta_t\right)\mathbf{N}$$

其中 $\eta = n_1 / n_2$，$\cos\theta_t = \sqrt{1 - \eta^2(1 - \cos^2\theta_i)}$。

当 $\sin^2\theta_t > 1$ 时发生**全内反射**，此时退化为镜面反射处理。

射线从玻璃内部穿出时需翻转法线方向，并将折射率比值取倒数（`n2/n1`）。

### 选做 2：抗锯齿 MSAA（+10%）

每像素在像素格内随机发射 `aa_samples` 条主射线（使用 `ti.random()` 生成亚像素偏移），将所有采样颜色取平均，有效消除物体边缘锯齿。通过 UI 滑条 `AA Samples`（范围 1~8）实时控制采样数。

---

## 核心算法原理

### 反射向量

$$\mathbf{R} = \mathbf{I} - 2(\mathbf{I} \cdot \mathbf{N})\mathbf{N}$$

### Whitted-Style 光线追踪流程

```
发射主光线
  ↓
场景求交（球体/平面解析求交）
  ↓
  ├── 未命中 → 返回背景色
  ├── 镜面材质 → 生成反射射线，throughput *= 反射率，继续迭代
  ├── 玻璃材质 → 折射/全内反射，throughput *= 透射率，继续迭代
  └── 漫反射材质 → 发射暗影射线，计算 Phong 光照，累加颜色，break
```
