#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.training.yolo_trainer import (
    run_yolo_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the LaRS compact5 YOLO baseline."
        )
    )

    parser.add_argument(
        "--mode",
        choices=("smoke", "baseline"),
        default="smoke",
    )

    parser.add_argument(
        "--project-config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "project.yaml"
        ),
    )

    parser.add_argument(
        "--yolo-config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "yolo_baseline.yaml"
        ),
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Override device: auto, mps, cpu, "
            "or a CUDA index such as 0."
        ),
    )

    parser.add_argument(
        "--name",
        default=None,
        help="Optional run-name override.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    run_dir = run_yolo_experiment(
        mode=args.mode,
        project_config_path=(
            args.project_config.resolve()
        ),
        yolo_config_path=(
            args.yolo_config.resolve()
        ),
        device_override=args.device,
        name_override=args.name,
    )

    print(
        f"Training run saved to: {run_dir}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())