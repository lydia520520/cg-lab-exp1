# Bezier Curve 绘制程序

基于 Taichi 框架实现的高性能 Bezier 曲线交互式绘制工具。


## 效果演示
![3D变换演示](https://github.com/lydia520520/cg-lab/blob/main/assets/Work3.gif)
(如长时间未加载出gif 可查看assets/Work3.gif)


## 功能特性

- **交互式绘制**：通过鼠标左键点击添加控制点
- **实时渲染**：使用 GPU 加速实现流畅的曲线绘制
- **高性能优化**：通过 GPU 并行计算和批量数据传输实现 60 FPS
- **控制点管理**：支持最多 100 个控制点
- **曲线可视化**：绿色曲线显示，红色控制点，灰色连接线

## 技术实现

### 核心算法
- 使用 **De Casteljau 算法**递归计算 Bezier 曲线上的点
- 曲线采样点数量：1000 个

### 性能优化策略

1. **GPU 缓冲区**：使用 `curve_points_field` 存储曲线坐标，减少内存传输
2. **并行绘制**：通过 `draw_curve_kernel` 在 GPU 上并行执行像素绘制
3. **批量传输**：一次性将所有曲线点数据从 CPU 传输到 GPU，避免频繁通信

### 技术栈
- **Taichi**：高性能并行计算框架
- **NumPy**：数值计算和数据处理
- **GPU 后端**：利用 GPU 加速计算

## 使用方法

### 环境要求
- Python 3.x
- Taichi
- NumPy

### 安装依赖
```bash
pip install taichi numpy
```

### 运行程序
```bash
python main.py
```

### 操作说明
- **鼠标左键**：点击窗口添加控制点
- **C 键**：清空所有控制点

