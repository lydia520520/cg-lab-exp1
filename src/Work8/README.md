# Work8: LBS 蒙皮

Author: 许艺珈

This folder contains the SMPL-based LBS pipeline required by `task8.txt`. The previous browser LBS demo is kept only as a supplemental toy visualization; the main submission is `main.py` and `src/smpl_lbs_pipeline.py`.

## Required Model File

The assignment requires the official `SMPL_NEUTRAL.pkl`, which is not included in this repository for license reasons. Put it in one of these locations:

- `work8/models/SMPL_NEUTRAL.pkl`
- a path passed with `python3 main.py --model-path /path/to/SMPL_NEUTRAL.pkl`
- a path/folder set in `SMPL_MODEL_PATH`

## Main Files

- `main.py` - command-line entry point for generating all required outputs.
- `src/smpl_lbs_pipeline.py` - SMPL loading, manual LBS, official forward comparison, visualization, summary, and animation.
- `outputs/MISSING_MODEL.txt` - generated when the official model file is absent.
- `tests/test_smpl_pipeline_contract.py` - tests for output contract and missing-model behavior.

## Generated Outputs

After adding `SMPL_NEUTRAL.pkl`, run:

```bash
python3 main.py
```

The script generates:

- `outputs/stage_a_template_weights.png`
- `outputs/stage_b_shaped_joints.png`
- `outputs/stage_c_pose_offsets.png`
- `outputs/stage_d_lbs_result.png`
- `outputs/comparison_grid.png`
- `outputs/all_joint_weights.png`
- `outputs/summary.txt`
- `outputs/lbs_pose_animation.gif`

## Implemented Requirements

- Loads SMPL using `smplx.create(..., model_type='smpl', gender='neutral')`.
- Prints/records vertex count, face count, joint count, and betas dimension in `summary.txt`.
- Explicitly computes and stores `v_template`, `v_shaped`, `J`, `v_posed`, and `verts`.
- Visualizes template weights, shaped mesh with joints, pose offsets, final LBS result, and a comparison grid.
- Includes optional all-joint dominant-weight visualization.
- Compares manual LBS vertices with official `model.forward(...)` vertices and records mean/max absolute error.
- Generates a simple joint animation GIF when the model file is available.

## Test

```bash
python3 -m unittest tests.test_smpl_pipeline_contract
```

The browser files (`index.html`, `src/lbs.js`) are supplemental and not the primary answer to the SMPL assignment.
