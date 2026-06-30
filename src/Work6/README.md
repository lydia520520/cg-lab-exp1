202311051007-许艺珈-人工智能
# 计算机图形学 实验六：可微渲染

## 效果演示（必做+选做）

![必做](https://github.com/lydia520520/cg-lab/blob/main/assets/Work61.png)
![选做](https://github.com/lydia520520/cg-lab/blob/main/assets/Work62.png)
(如长时间未加载出png 可查看assets/Work61.png)
(如长时间未加载出png 可查看assets/Work62.png)



## 保留文件

- `lab6_core.py`：公共渲染与优化逻辑
- `lab6_required.py`：必做，左侧 `Target`、右侧 `Current`
- `lab6_extra.py`：选做，左侧 `Target`、中间 `Lambert`、右侧 `Leaky`
- `README.md`：运行说明

## 实验说明

- 前向渲染采用球体的 Ray Casting
- 优化目标是通过 Taichi AutoDiff 反向更新光源位置
- 必做使用 Leaky Lambertian，让背光区域仍保留梯度
- 选做对比标准 Lambertian 与 Leaky Lambertian 在优化上的差别

## 环境准备

在项目根目录执行：

```bash
cd /Users/lydia/Documents/github/cg-lab
uv sync
```

## 运行命令

### 必做

如果你已经在 `src/work6` 目录：

```bash
uv run python lab6_required.py --arch cpu --resolution 64 --iters 30
```

如果你在项目根目录：

```bash
uv run python src/work6/lab6_required.py --arch cpu --resolution 64 --iters 30
```

说明：

- 默认弹出实时窗口
- 左边显示 `Target`
- 右边显示 `Current`
- 关闭窗口后程序结束

### 选做

如果你已经在 `src/work6` 目录：

```bash
uv run python lab6_extra.py --arch cpu --resolution 64 --iters 30
```

如果你在项目根目录：

```bash
uv run python src/work6/lab6_extra.py --arch cpu --resolution 64 --iters 30
```

说明：

- 默认弹出实时窗口
- 左边显示 `Target`
- 中间显示 `Lambert`
- 右边显示 `Leaky`
- 关闭窗口后程序结束

## 无界面模式

如果只想测试是否能跑通，可以加 `--headless`：

```bash
uv run python src/work6/lab6_required.py --arch cpu --resolution 32 --iters 4 --headless
uv run python src/work6/lab6_extra.py --arch cpu --resolution 32 --iters 4 --headless
```

## 输出结果

运行时会自动创建 `outputs/` 目录，并写入：

- `target.pgm`
- `prediction_final.pgm`
- `history.csv`
- `summary.txt`
- `comparison.txt`

`outputs/` 不再预置保存，需要时运行脚本会重新生成。
