#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.validate_yolo_dataset import (
    validate_yolo_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the processed LaRS YOLO dataset."
        )
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
        "--report",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "metrics"
            / "yolo"
            / "dataset_verification.json"
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_config = yaml.safe_load(
        args.project_config.read_text(
            encoding="utf-8"
        )
    )

    dataset_root = (
        PROJECT_ROOT
        / project_config["dataset"]["yolo_root"]
    )

    report = validate_yolo_dataset(
        dataset_yaml=dataset_root / "dataset.yaml",
        expected_class_names=[
            str(name)
            for name in project_config["classes"]
        ],
        import_summary_path=(
            dataset_root / "import_summary.json"
        ),
    )

    print(f"Dataset root: {report.dataset_root}")
    print(f"Classes: {report.class_names}")

    for split in report.splits:
        print(f"\n{split.split.upper()}")
        print(f"  Images:       {split.image_count}")
        print(f"  Labels:       {split.label_count}")
        print(
            f"  Empty labels: "
            f"{split.empty_label_count}"
        )
        print(f"  Objects:      {split.object_count}")

        for class_name, count in (
            split.class_counts.items()
        ):
            print(
                f"    {class_name:<20} "
                f"{count:>6}"
            )

    if report.warnings:
        print("\nWarnings:")

        for warning in report.warnings:
            print(f"  - {warning}")

    if report.errors:
        print("\nErrors:")

        for error in report.errors:
            print(f"  - {error}")

    args.report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.report.write_text(
        json.dumps(
            report.to_dict(),
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nReport: {args.report}")

    if report.passed:
        print("VALIDATION PASSED")
        return 0

    print("VALIDATION FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())