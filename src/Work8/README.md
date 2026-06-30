202311051007-许艺珈-人工智能
# Work8: LBS 蒙皮

## 效果演示(必做+选做)

![选做](https://github.com/lydia520520/cg-lab/blob/main/assets/Work8.gif)
(如长时间未加载出gif 可查看assets/Work8.gif 和 Work8/outputs)


## 保留文件

- `main.py`：运行入口，默认生成结果后弹出预览窗口
- `src/smpl_lbs_pipeline.py`：SMPL 加载、手写 LBS、中间量可视化与结果导出
- `src/__init__.py`：包初始化文件
- `requirements.txt`：本实验使用的 Python 依赖
- `SMPL_NEUTRAL.pkl`：官方 SMPL 中性人体模型文件
- `outputs/`：当前已生成的成果图、GIF 和摘要

## 运行方式

在 `Work8` 目录下运行：

```bash
uv run python main.py
```

如果只想生成文件、不弹出预览窗口：

```bash
uv run python main.py --no-show
```

如果模型文件不在当前目录，可以手动指定：

```bash
uv run python main.py --model-path /path/to/SMPL_NEUTRAL.pkl
```

## 当前输出

运行后会生成或更新以下文件：

- `outputs/stage_a_template_weights.png`
- `outputs/stage_b_shaped_joints.png`
- `outputs/stage_c_pose_offsets.png`
- `outputs/stage_d_lbs_result.png`
- `outputs/comparison_grid.png`
- `outputs/all_joint_weights.png`
- `outputs/lbs_pose_animation.gif`
- `outputs/summary.txt`

## 实现内容

- 使用 `smplx.create(..., model_type="smpl", gender="neutral")` 加载 SMPL
- 明确计算并区分 `v_template`、`v_shaped`、`J`、`v_posed`、`verts`
- 可视化模板权重、形状校正与关节、姿态校正、最终 LBS 结果
- 生成总对比图和全关节主导权重图
- 将手写 LBS 与官方 `model.forward(...)` 结果进行误差对比并写入 `summary.txt`
- 生成简单姿态动画 `lbs_pose_animation.gif`

## 说明

- 代码已兼容一部分旧版 `SMPL_NEUTRAL.pkl` 的 `chumpy` 依赖
- 如果当前环境无法打开图形窗口，程序仍会正常把结果保存到 `outputs/`
