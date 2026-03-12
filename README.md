# CG-Lab 实验0：Taichi 引力粒子群

## 效果演示

![粒子群引力模拟演示](https://github.com/lydia520520/cg-lab-exp1/blob/main/assets/Work0.GIF)
(如长时间未加载出gif 可查看assets/Work0.gif)

## 项目简介

这是一个基于 Taichi 的计算机图形学实验项目，实现了粒子群在鼠标引力作用下的动态物理模拟。项目使用 GPU 并行计算，能够高效处理数千个粒子的实时物理交互。

## 项目架构

```
cg-lab-exp1/
├── src/
│   └── Work0/           # 实验0：引力粒子群
│       ├── __init__.py  # 模块初始化
│       ├── config.py    # 物理参数配置
│       ├── main.py      # 主程序入口
│       └── physics.py   # 物理引擎核心
├── main.py              # 项目启动入口
├── pyproject.toml       # 项目依赖配置
└── README.md           # 项目文档
```

## 技术栈

- **编程语言**: Python 3.12+
- **图形框架**: Taichi 1.7.4+
- **物理引擎**: 自定义 GPU 并行物理计算
- **渲染**: Taichi GUI 实时渲染

## 功能特性

### 物理模拟
- **粒子系统**: 支持数千个粒子的实时模拟
- **引力交互**: 鼠标位置产生引力场，吸引粒子跟随移动
- **物理效果**: 包含空气阻力、边界反弹等物理特性
- **GPU 加速**: 使用 Taichi 框架实现 GPU 并行计算

### 交互功能
- **鼠标控制**: 鼠标移动产生引力，吸引粒子跟随
- **实时渲染**: 60FPS 的流畅粒子动画
- **窗口适配**: 支持窗口大小动态调整

## 快速开始

### 环境要求

- Python 3.12 或更高版本
- 支持 CUDA 的 NVIDIA GPU（推荐）或兼容的 GPU

### 安装依赖

```bash
pip install taichi>=1.7.4
```

### 运行项目

```bash
python -m src.Work0.main
```

## 配置参数

在 `src/Work0/config.py` 中可以调整以下参数：

```python
# 物理系统参数
NUM_PARTICLES = 2000     # 粒子总数
GRAVITY_STRENGTH = 0.001 # 鼠标引力强度
DRAG_COEF = 0.98         # 空气阻力系数
BOUNCE_COEF = -0.8       # 边界反弹能量损耗

# 渲染系统参数
WINDOW_RES = (800, 600)  # 窗口分辨率
PARTICLE_RADIUS = 1.5    # 粒子绘制半径
PARTICLE_COLOR = 0x00BFFF # 粒子颜色（天蓝色）
```

## 代码架构

1. **config.py** - 参数配置
   - 集中管理所有物理和渲染参数
   - 便于性能调优和效果调整

2. **physics.py** - 物理引擎
   - 定义粒子位置和速度场
   - 实现 GPU 并行的物理更新内核
   - 包含引力计算、阻力模拟、边界碰撞检测

3. **main.py** - 主程序
   - 初始化 Taichi 和 GUI
   - 处理鼠标输入和渲染循环
   - 协调物理计算和图形渲染

## 性能优化

### GPU 并行化
项目充分利用 Taichi 的 GPU 并行计算能力，每个粒子的物理计算都是独立并行的，确保即使处理数千个粒子也能保持流畅性能。

### 内存优化
使用 Taichi 的数据结构直接在 GPU 显存中存储粒子状态，减少 CPU-GPU 数据传输开销。


## 许可证

本项目仅供学习使用，遵循 MIT 许可证。
