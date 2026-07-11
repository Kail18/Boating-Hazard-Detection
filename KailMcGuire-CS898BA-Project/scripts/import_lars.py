#!/usr/bin/env python3
"""Import LaRS train/validation data and convert annotations to YOLO format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.lars_to_yolo import PROFILES, convert_split, write_dataset_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert LaRS COCO-panoptic train/val annotations into a YOLO "
            "object-detection dataset."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "lars",
        help="Extracted LaRS root (default: data/raw/lars)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "lars_yolo",
        help="YOLO dataset output root",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="compact5",
        help="compact5 merges sparse native labels; native8 preserves all classes",
    )
    parser.add_argument(
        "--transfer-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="How images are placed in the model-ready dataset",
    )
    parser.add_argument(
        "--min-box-area",
        type=float,
        default=4.0,
        help="Skip boxes smaller than this many source-image pixels",
    )
    return parser.parse_args()


def locate_split(source_root: Path, split: str) -> tuple[Path, Path]:
    """Locate a split across common LaRS extraction layouts."""
    candidate_roots = (
        source_root / split,
        source_root / "split" / split,
        source_root / "LaRS" / split,
        source_root / "LaRS" / "split" / split,
    )
    annotation_names = (
        "panoptic_annotations.json",
        "annotations.json",
    )

    for split_root in candidate_roots:
        image_dir = split_root / "images"
        for annotation_name in annotation_names:
            annotation_path = split_root / annotation_name
            if image_dir.is_dir() and annotation_path.is_file():
                return image_dir, annotation_path

    attempted = "\n".join(f"  - {path}" for path in candidate_roots)
    raise FileNotFoundError(
        f"Could not locate LaRS '{split}' split beneath {source_root}. "
        f"Expected an images directory and panoptic_annotations.json in one of:\n"
        f"{attempted}"
    )


def main() -> int:
    args = parse_args()
    profile = PROFILES[args.profile]
    summaries: list[dict[str, object]] = []

    for split in ("train", "val"):
        image_dir, annotation_path = locate_split(args.source, split)
        print(f"Converting {split}: {image_dir}")
        summary = convert_split(
            split_name=split,
            image_dir=image_dir,
            annotation_path=annotation_path,
            output_root=args.output,
            profile=profile,
            transfer_mode=args.transfer_mode,
            min_box_area=args.min_box_area,
        )
        summaries.append(summary)
        print(
            f"  exported {summary['images_exported']} images and "
            f"{summary['objects_exported']} objects"
        )

    dataset_yaml = write_dataset_yaml(args.output, profile.class_names)
    report = {
        "dataset": "LaRS v1.0.0",
        "profile": profile.name,
        "class_names": list(profile.class_names),
        "dataset_yaml": str(dataset_yaml),
        "splits": summaries,
    }
    report_path = args.output / "import_summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Dataset YAML: {dataset_yaml}")
    print(f"Import summary: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
