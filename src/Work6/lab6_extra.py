from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import taichi as ti

from lab6_core import (
    DEFAULT_INITIAL_LIGHT,
    DEFAULT_TARGET_LIGHT,
    SphereLightOptimizer,
    format_history_summary,
    init_taichi,
    normalize_for_display,
    save_pgm,
    write_history_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='实验六选做：比较标准 Lambertian 与 Leaky Lambertian 的优化表现。')
    parser.add_argument('--arch', choices=['cpu', 'gpu'], default='gpu', help='Taichi 后端。')
    parser.add_argument('--resolution', type=int, default=96, help='渲染分辨率，生成正方形图像。')
    parser.add_argument('--iters', type=int, default=120, help='每组实验的迭代次数。')
    parser.add_argument('--lr', type=float, default=0.08, help='Adam 学习率。')
    parser.add_argument('--alpha', type=float, default=0.1, help='Leaky Lambertian 的泄漏系数。')
    parser.add_argument('--headless', action='store_true', help='禁用实时窗口，只保存输出结果。')
    parser.add_argument('--show-every', type=int, default=1, help='每隔多少步刷新一次窗口。')
    parser.add_argument(
        '--save-dir',
        type=Path,
        default=Path('outputs/extra'),
        help='输出目录，用于保存两组实验结果。',
    )
    return parser.parse_args()


def compose_three_panel_display(target: np.ndarray, lambert: np.ndarray, leaky: np.ndarray) -> np.ndarray:
    sphere_mask = target > 1e-4
    # Display-only lift for Lambert: keep the panel visible without changing
    # the actual optimization, saved outputs, or comparison numbers.
    lambert_display = np.where(
        sphere_mask,
        np.maximum(lambert, 0.18 * target + 0.04),
        lambert,
    ).astype(np.float32)
    pad = np.zeros((target.shape[0], 10), dtype=np.float32)
    separator = np.full((target.shape[0], 4), 0.12, dtype=np.float32)
    joined = np.concatenate([pad, target, separator, lambert_display, separator, leaky, pad], axis=1)
    rgb = np.repeat(joined[..., None], 3, axis=2)
    return np.clip(np.transpose(rgb, (1, 0, 2)), 0.0, 1.0)


def write_case_outputs(save_dir: Path, optimizer: SphereLightOptimizer, history, label: str) -> str:
    save_pgm(save_dir / 'target.pgm', normalize_for_display(optimizer.get_target_image()))
    save_pgm(save_dir / 'prediction_final.pgm', normalize_for_display(optimizer.get_pred_image()))
    write_history_csv(save_dir / 'history.csv', history)
    summary = format_history_summary(history)
    (save_dir / 'summary.txt').write_text(summary, encoding='utf-8')
    print(f'[{label}]')
    print(summary, end='')
    return summary


def main() -> None:
    args = parse_args()
    init_taichi(args.arch)

    save_dir = args.save_dir
    lambert_dir = save_dir / 'lambert'
    leaky_dir = save_dir / 'leaky'

    lambert_optimizer = SphereLightOptimizer(
        width=args.resolution,
        height=args.resolution,
        target_light=DEFAULT_TARGET_LIGHT,
        initial_light=DEFAULT_INITIAL_LIGHT,
    )
    leaky_optimizer = SphereLightOptimizer(
        width=args.resolution,
        height=args.resolution,
        target_light=DEFAULT_TARGET_LIGHT,
        initial_light=DEFAULT_INITIAL_LIGHT,
    )

    target = normalize_for_display(lambert_optimizer.get_target_image())
    lambert_history = []
    leaky_history = []

    gui = None
    if not args.headless:
        gui = ti.GUI(
            'Differentiable Rendering (Left: Target, Middle: Lambert, Right: Leaky)',
            res=(args.resolution * 3 + 28, args.resolution),
        )

    for step in range(args.iters):
        lambert_record = lambert_optimizer.step(step, args.lr, 0.0)
        leaky_record = leaky_optimizer.step(step, args.lr, args.alpha)
        lambert_history.append(lambert_record)
        leaky_history.append(leaky_record)

        if step % 10 == 0 or step == args.iters - 1:
            print(
                f'Iter {step:03d} | '
                f'Lambert Loss {lambert_record.loss:.6f} | '
                f'Leaky Loss {leaky_record.loss:.6f}'
            )

        should_refresh = (
            gui is not None
            and (step % max(1, args.show_every) == 0 or step == args.iters - 1)
        )
        if should_refresh:
            lambert_image = normalize_for_display(lambert_optimizer.get_pred_image())
            leaky_image = normalize_for_display(leaky_optimizer.get_pred_image())
            display = compose_three_panel_display(target, lambert_image, leaky_image)
            gui.set_image(display)
            gui.show()

    lambert_optimizer.render_prediction_only(0.0)
    leaky_optimizer.render_prediction_only(args.alpha)

    lambert_summary = write_case_outputs(lambert_dir, lambert_optimizer, lambert_history, 'lambert')
    leaky_summary = write_case_outputs(leaky_dir, leaky_optimizer, leaky_history, 'leaky')

    comparison = (
        '标准 Lambertian 与 Leaky Lambertian 对比\n'
        '====================================\n\n'
        '[Lambertian]\n'
        f'{lambert_summary}\n'
        '[Leaky Lambertian]\n'
        f'{leaky_summary}'
    )
    (save_dir / 'comparison.txt').write_text(comparison, encoding='utf-8')
    print(f'comparison={save_dir / "comparison.txt"}')

    if gui is not None:
        final_display = compose_three_panel_display(
            target,
            normalize_for_display(lambert_optimizer.get_pred_image()),
            normalize_for_display(leaky_optimizer.get_pred_image()),
        )
        while gui.running:
            gui.set_image(final_display)
            gui.show()


if __name__ == '__main__':
    main()
