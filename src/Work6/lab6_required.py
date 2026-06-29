from __future__ import annotations

import argparse
from pathlib import Path

import taichi as ti

from lab6_core import (
    DEFAULT_INITIAL_LIGHT,
    DEFAULT_TARGET_LIGHT,
    SphereLightOptimizer,
    compose_display_image,
    format_history_summary,
    init_taichi,
    normalize_for_display,
    save_pgm,
    write_history_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='实验六必做：基于 Taichi 的可微渲染光源反演。')
    parser.add_argument('--arch', choices=['cpu', 'gpu'], default='gpu', help='Taichi 后端。')
    parser.add_argument('--resolution', type=int, default=96, help='渲染分辨率，生成正方形图像。')
    parser.add_argument('--iters', type=int, default=120, help='梯度下降迭代次数。')
    parser.add_argument('--lr', type=float, default=0.08, help='Adam 学习率。')
    parser.add_argument('--alpha', type=float, default=0.1, help='Leaky Lambertian 泄漏系数。')
    parser.add_argument('--headless', action='store_true', help='禁用实时窗口，只保存输出结果。')
    parser.add_argument('--show-every', type=int, default=1, help='每隔多少步刷新一次窗口。')
    parser.add_argument(
        '--save-dir',
        type=Path,
        default=Path('outputs/required'),
        help='输出目录，用于保存目标图、预测图和 loss 历史。',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_taichi(args.arch)

    optimizer = SphereLightOptimizer(
        width=args.resolution,
        height=args.resolution,
        target_light=DEFAULT_TARGET_LIGHT,
        initial_light=DEFAULT_INITIAL_LIGHT,
    )

    target = normalize_for_display(optimizer.get_target_image())
    history = []

    gui = None
    if not args.headless:
        gui = ti.GUI(
            'Differentiable Rendering',
            res=(args.resolution * 2 + 2, args.resolution),
        )

    for step in range(args.iters):
        record = optimizer.step(step, args.lr, args.alpha)
        history.append(record)
        if step % 10 == 0 or step == args.iters - 1:
            print(
                f'Iter {step:03d} | Loss {record.loss:.6f} | '
                f'Light ({record.light_x:.3f}, {record.light_y:.3f}, {record.light_z:.3f})'
            )

        should_refresh = (
            gui is not None
            and (step % max(1, args.show_every) == 0 or step == args.iters - 1)
        )
        if should_refresh:
            pred = normalize_for_display(optimizer.get_pred_image())
            display = compose_display_image(target, pred)
            gui.set_image(display)
            gui.show()

    optimizer.render_prediction_only(args.alpha)

    save_dir = args.save_dir
    pred = normalize_for_display(optimizer.get_pred_image())
    save_pgm(save_dir / 'target.pgm', target)
    save_pgm(save_dir / 'prediction_final.pgm', pred)
    write_history_csv(save_dir / 'history.csv', history)

    summary = format_history_summary(history)
    (save_dir / 'summary.txt').write_text(summary, encoding='utf-8')
    print('[lab6_required] done')
    print(summary, end='')
    print(f'outputs={save_dir}')

    if gui is not None:
        final_display = compose_display_image(target, pred)
        while gui.running:
            gui.set_image(final_display)
            gui.show()


if __name__ == '__main__':
    main()
