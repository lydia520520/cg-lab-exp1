202311051007-许艺珈-人工智能
# 计算机图形学 实验七：质点弹簧模型

## 效果演示（必做+选做）

![必做](https://github.com/lydia520520/cg-lab/blob/main/assets/Work71.gif)
![选做](https://github.com/lydia520520/cg-lab/blob/main/assets/Work72.gif)
(如长时间未加载出gif 可查看assets/Work71.gif和assets/Work72.gif)

## 保留文件

- `main.py`：必做部分，一键运行 Taichi 可视化窗口
- `main_extra.py`：选做部分，一键运行可视化窗口
- `taichi_mass_spring.py`：布料模拟核心逻辑
- `README.md`：运行说明

## 文件说明

### 必做：`main.py`

- 只保留结构弹簧
- 不启用剪切弹簧、弯曲弹簧、球体碰撞
- 提供可视化窗口与控制面板

### 选做：`main_extra.py`

- 启用剪切弹簧
- 启用弯曲弹簧
- 启用球体碰撞
- 提供更适合展示的窗口标题、相机、配色和光照

### 核心：`taichi_mass_spring.py`

- 质点位置、速度、受力场
- 多 kernel 初始化
- 三种积分器：
  - `explicit`
  - `semi_implicit`
  - `implicit`
- 选做内容所需的剪切弹簧、弯曲弹簧、球碰撞

## 环境准备

在项目根目录执行：

```bash
cd /Users/lydia/Documents/github/cg-lab
uv sync
```

## 一键运行

### 如果你已经在 `src/Work7` 目录

必做：

```bash
uv run python main.py
```

选做：

```bash
uv run python main_extra.py
```

### 如果你在项目根目录

必做：

```bash
uv run python src/Work7/main.py
```

选做：

```bash
uv run python src/Work7/main_extra.py
```

## 运行效果

- 程序会直接弹出 Taichi 可视化窗口
- 左上角控制面板可切换积分器
- 可以暂停、继续、重置
- 可以调节重力和刚度
- 按 `R` 可以快速重置布料

## 自动化启动测试

必做：

```bash
uv run python src/Work7/main.py --max-frames 5
```

选做：

```bash
uv run python src/Work7/main_extra.py --max-frames 5
```

`--max-frames 5` 会在渲染 5 帧后自动退出，适合快速验证入口是否正常。
