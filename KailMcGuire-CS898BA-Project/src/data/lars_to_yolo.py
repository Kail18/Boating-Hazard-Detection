"""Convert LaRS COCO-panoptic annotations into YOLO detection labels.

LaRS provides panoptic annotations with COCO-style ``segments_info`` entries.
Each segment includes a bounding box, so a detection baseline can be produced
without reconstructing boxes from the mask pixels.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

NATIVE8_NAMES = (
    "boat_ship",
    "row_boat",
    "buoy",
    "float",
    "paddle_board",
    "swimmer",
    "animal",
    "other",
)

COMPACT5_NAMES = (
    "vessel",
    "buoy",
    "swimmer",
    "paddle_board",
    "floating_obstacle",
)

# Canonicalized LaRS category name -> class name for each profile.
NATIVE8_MAPPING = {
    "boat ship": "boat_ship",
    "boat": "boat_ship",
    "ship": "boat_ship",
    "row boat": "row_boat",
    "row boats": "row_boat",
    "buoy": "buoy",
    "float": "float",
    "paddle board": "paddle_board",
    "paddleboard": "paddle_board",
    "swimmer": "swimmer",
    "animal": "animal",
    "other": "other",
}

COMPACT5_MAPPING = {
    "boat ship": "vessel",
    "boat": "vessel",
    "ship": "vessel",
    "row boat": "vessel",
    "row boats": "vessel",
    "buoy": "buoy",
    "swimmer": "swimmer",
    "paddle board": "paddle_board",
    "paddleboard": "paddle_board",
    "float": "floating_obstacle",
    "animal": "floating_obstacle",
    "other": "floating_obstacle",
}


@dataclass(frozen=True)
class ConversionProfile:
    """Mapping and output names for a LaRS-to-YOLO conversion."""

    name: str
    class_names: tuple[str, ...]
    category_mapping: Mapping[str, str]


PROFILES = {
    "compact5": ConversionProfile("compact5", COMPACT5_NAMES, COMPACT5_MAPPING),
    "native8": ConversionProfile("native8", NATIVE8_NAMES, NATIVE8_MAPPING),
}


def canonicalize_category(name: str) -> str:
    """Normalize category spelling and separators for robust matching."""
    normalized = name.strip().lower()
    for token in ("/", "_", "-"):
        normalized = normalized.replace(token, " ")
    return " ".join(normalized.split())


def yolo_bbox(
    bbox_xywh: Iterable[float], image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    """Convert an absolute COCO ``[x, y, width, height]`` box to YOLO form."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive")

    x, y, width, height = (float(value) for value in bbox_xywh)
    if width <= 0 or height <= 0:
        raise ValueError(f"Bounding-box dimensions must be positive: {bbox_xywh}")

    # Clamp boxes to the image to prevent invalid normalized values caused by
    # small annotation rounding errors.
    x1 = min(max(x, 0.0), float(image_width))
    y1 = min(max(y, 0.0), float(image_height))
    x2 = min(max(x + width, 0.0), float(image_width))
    y2 = min(max(y + height, 0.0), float(image_height))

    clamped_width = x2 - x1
    clamped_height = y2 - y1
    if clamped_width <= 0 or clamped_height <= 0:
        raise ValueError(f"Bounding box falls outside image bounds: {bbox_xywh}")

    center_x = (x1 + x2) / 2.0 / image_width
    center_y = (y1 + y2) / 2.0 / image_height
    norm_width = clamped_width / image_width
    norm_height = clamped_height / image_height
    return center_x, center_y, norm_width, norm_height


def _transfer_file(source: Path, destination: Path, mode: str) -> str:
    """Transfer an image with hardlink/copy/symlink behavior.

    Hardlink mode automatically falls back to copying if the files are on
    different volumes or the platform rejects hard links.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()

    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    if mode == "symlink":
        destination.symlink_to(os.path.relpath(source, destination.parent))
        return "symlink"
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copy_fallback"
    raise ValueError(f"Unsupported transfer mode: {mode}")


def _load_annotation(annotation_path: Path) -> dict:
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation JSON not found: {annotation_path}")
    with annotation_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    for required_key in ("images", "annotations", "categories"):
        if required_key not in data:
            raise ValueError(
                f"{annotation_path} is missing required COCO-panoptic key: "
                f"{required_key}"
            )
    return data


def convert_split(
    *,
    split_name: str,
    image_dir: Path,
    annotation_path: Path,
    output_root: Path,
    profile: ConversionProfile,
    transfer_mode: str = "hardlink",
    min_box_area: float = 4.0,
) -> dict[str, object]:
    """Convert one LaRS split into YOLO image and label directories."""
    data = _load_annotation(annotation_path)

    categories_by_id = {
        int(category["id"]): canonicalize_category(str(category["name"]))
        for category in data["categories"]
    }
    images_by_id = {int(image["id"]): image for image in data["images"]}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in data["annotations"]:
        annotations_by_image[int(annotation["image_id"])].extend(
            annotation.get("segments_info", [])
        )

    class_to_id = {name: index for index, name in enumerate(profile.class_names)}
    output_image_dir = output_root / "images" / split_name
    output_label_dir = output_root / "labels" / split_name
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    class_counts: Counter[str] = Counter()
    skipped_categories: Counter[str] = Counter()
    transfer_counts: Counter[str] = Counter()
    missing_images: list[str] = []
    invalid_boxes = 0
    images_with_labels = 0
    empty_label_images = 0

    for image_id, image_record in sorted(images_by_id.items()):
        file_name = str(image_record["file_name"])
        source_image = image_dir / file_name
        if not source_image.exists():
            # Some archives store file_name with a nested prefix while images
            # have been extracted directly into the images directory.
            source_image = image_dir / Path(file_name).name
        if not source_image.exists():
            missing_images.append(file_name)
            continue
        if source_image.suffix.lower() not in VALID_IMAGE_SUFFIXES:
            continue

        width = int(image_record["width"])
        height = int(image_record["height"])
        label_lines: list[str] = []

        for segment in annotations_by_image.get(image_id, []):
            category_name = categories_by_id.get(
                int(segment["category_id"]), f"unknown_{segment['category_id']}"
            )
            output_class = profile.category_mapping.get(category_name)
            if output_class is None:
                skipped_categories[category_name] += 1
                continue

            bbox = segment.get("bbox")
            if not bbox or len(bbox) != 4:
                invalid_boxes += 1
                continue
            if float(bbox[2]) * float(bbox[3]) < min_box_area:
                invalid_boxes += 1
                continue

            try:
                center_x, center_y, box_width, box_height = yolo_bbox(
                    bbox, width, height
                )
            except (TypeError, ValueError):
                invalid_boxes += 1
                continue

            class_id = class_to_id[output_class]
            label_lines.append(
                f"{class_id} {center_x:.6f} {center_y:.6f} "
                f"{box_width:.6f} {box_height:.6f}"
            )
            class_counts[output_class] += 1

        destination_image = output_image_dir / source_image.name
        transfer_result = _transfer_file(source_image, destination_image, transfer_mode)
        transfer_counts[transfer_result] += 1

        destination_label = output_label_dir / f"{source_image.stem}.txt"
        destination_label.write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )
        if label_lines:
            images_with_labels += 1
        else:
            empty_label_images += 1

    return {
        "split": split_name,
        "profile": profile.name,
        "images_declared": len(images_by_id),
        "images_exported": sum(transfer_counts.values()),
        "images_with_labels": images_with_labels,
        "empty_label_images": empty_label_images,
        "objects_exported": sum(class_counts.values()),
        "class_counts": dict(class_counts),
        "skipped_categories": dict(skipped_categories),
        "invalid_boxes": invalid_boxes,
        "missing_images": missing_images,
        "transfer_counts": dict(transfer_counts),
    }


def write_dataset_yaml(output_root: Path, class_names: Iterable[str]) -> Path:
    """Write an Ultralytics-compatible dataset YAML file."""
    class_names = tuple(class_names)
    lines = [
        f"path: {output_root.resolve()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(class_names))
    yaml_path = output_root / "dataset.yaml"
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path
