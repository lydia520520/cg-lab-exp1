from __future__ import annotations

import argparse
from pathlib import Path

from src.smpl_lbs_pipeline import ModelFileMissing, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SMPL LBS visualizations for Work8.")
    parser.add_argument("--model-path", action="append", default=[], help="SMPL_NEUTRAL.pkl file or folder containing it.")
    parser.add_argument("--output-dir", default="outputs", help="Output directory relative to work8.")
    parser.add_argument("--no-show", action="store_true", help="Save outputs only and skip the preview window.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    output_dir = root / args.output_dir
    try:
        default_candidates = [root / "SMPL_NEUTRAL.pkl"]
        run_pipeline(output_dir=output_dir, model_candidates=[*default_candidates, *args.model_path], show_window=not args.no_show)
    except ModelFileMissing as exc:
        print(exc)
        raise SystemExit(2)
    print(f"wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
