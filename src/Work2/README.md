202311051007-许艺珈-人工智能
# Work2: 3D 变换演示

## 效果演示(必做+选做)

![3D变换演示](https://github.com/lydia520520/cg-lab/blob/main/assets/Work2.gif)
![选做](https://github.com/lydia520520/cg-lab/blob/main/assets/Work22.gif)
(如长时间未加载出gif 可查看assets/Work2.gif和assets/Work22.gif)

## 项目简介

这是一个使用 Taichi 实现的 3D 坐标变换演示程序，展示了计算机图形学中的模型变换、视图变换和透视投影变换的完整流程。

## 功能特性

- **3D 变换**：实现了完整的 MVP（Model-View-Projection）变换
- **实时渲染**：使用 Taichi GUI 实时显示变换结果
- **交互控制**：支持键盘控制三角形旋转
- **教学演示**：清晰展示了矩阵变换的数学原理

## 技术实现

### 核心功能

1. **模型变换**：绕 Z 轴旋转
2. **透视投影**：将 3D 坐标投影到 2D 屏幕


### 关键矩阵

- **模型矩阵** (`get_model_matrix`)：实现绕 Z 轴的旋转变换
- **视图矩阵** (`get_view_matrix`)：实现相机位置的调整
- **投影矩阵** (`get_projection_matrix`)：实现透视投影效果
- **MVP 矩阵**：模型、视图、投影矩阵的组合

## 运行方法

### 环境要求

- Python 3.12+
- Taichi 1.7.4+

### 必做启动命令

在项目根目录执行：

```bash
uv run python -m src.Work2.main
```

### 选做启动命令

选做部分新增了独立入口 `main_extra.py`，会弹出一个可视化窗口，展示 3D 立方体的透视投影，以及两个旋转姿态 `R0 -> Rt -> R1` 之间的插值动画：

```bash
uv run python -m src.Work2.main_extra
```

如果只想做快速验证，可以自动运行少量帧后退出：

```bash
uv run python -m src.Work2.main_extra --headless --max-frames 10
```

### 交互操作

- **A 键**：逆时针旋转三角形
- **D 键**：顺时针旋转三角形
- **ESC 键**：退出程序

## 代码结构

```
src/Work2/
├── main.py         # 必做：三角形 MVP 变换
├── main_extra.py   # 选做：3D 立方体旋转插值
└── README.md       # 项目说明文档
```

## 运行效果

程序运行后会显示一个 700x700 的窗口，包含一个彩色三角形。通过 A/D 键可以控制三角形绕 Z 轴旋转，观察 3D 变换的效果。

选做窗口会显示三个立方体：左侧为起始姿态 `R0`，右侧为目标姿态 `R1`，中间为通过旋转插值实时变化的 `Rt`。

## 许可证

本项目仅供学习使用。
