"""Validation helpers for the processed LaRS YOLO dataset."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


VALID_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

BOX_TOLERANCE = 1e-5


@dataclass
class SplitSummary:
    """Store validation statistics for one dataset split."""

    split: str
    image_count: int
    label_count: int
    empty_label_count: int
    object_count: int
    class_counts: dict[str, int]


@dataclass
class ValidationReport:
    """Store the complete result of YOLO dataset validation."""

    dataset_yaml: str
    dataset_root: str
    class_names: list[str]
    splits: list[SplitSummary]
    errors: list[str]
    warnings: list[str]

    @property
    def passed(self) -> bool:
        """Return True when the validator found no errors."""
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert the validation report into a JSON-compatible dictionary."""
        return {
            "dataset_yaml": self.dataset_yaml,
            "dataset_root": self.dataset_root,
            "class_names": self.class_names,
            "splits": [
                asdict(split_summary)
                for split_summary in self.splits
            ],
            "errors": self.errors,
            "warnings": self.warnings,
            "passed": self.passed,
        }


def summarize_image_directory(directory: Path) -> dict[str, object]:
    """Return image totals and extension counts for a directory tree."""
    if not directory.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Expected a directory but received: {directory}"
        )

    images = [
        path
        for path in directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in VALID_IMAGE_SUFFIXES
        )
    ]

    extension_counts = Counter(
        path.suffix.lower()
        for path in images
    )

    return {
        "directory": str(directory),
        "image_count": len(images),
        "extensions": dict(
            sorted(extension_counts.items())
        ),
    }


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML file and require a dictionary at its top level."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a YAML mapping in: {path}"
        )

    return data


def _normalize_class_names(raw_names: Any) -> list[str]:
    """Convert YOLO class-name configuration into an ordered list."""
    if isinstance(raw_names, list):
        class_names = [
            str(class_name)
            for class_name in raw_names
        ]

        if not class_names:
            raise ValueError(
                "dataset.yaml contains an empty class-name list."
            )

        return class_names

    if isinstance(raw_names, dict):
        normalized_names = {
            int(class_id): str(class_name)
            for class_id, class_name in raw_names.items()
        }

        expected_ids = list(
            range(len(normalized_names))
        )

        actual_ids = sorted(normalized_names)

        if actual_ids != expected_ids:
            raise ValueError(
                "Class IDs in dataset.yaml must begin at 0 "
                "and remain contiguous. "
                f"Found IDs: {actual_ids}"
            )

        return [
            normalized_names[class_id]
            for class_id in expected_ids
        ]

    raise ValueError(
        "dataset.yaml must define 'names' as either "
        "a list or dictionary."
    )


def _resolve_dataset_root(
    dataset_yaml: Path,
    config: dict[str, Any],
) -> Path:
    """Resolve the dataset root specified by the YOLO YAML file."""
    configured_root = config.get("path")

    if configured_root in (None, ""):
        return dataset_yaml.parent.resolve()

    root = Path(
        str(configured_root)
    ).expanduser()

    if not root.is_absolute():
        root = dataset_yaml.parent / root

    return root.resolve()


def _resolve_split_path(
    dataset_root: Path,
    split_value: Any,
) -> Path:
    """Resolve a train or validation image path against the dataset root."""
    split_path = Path(
        str(split_value)
    ).expanduser()

    if split_path.is_absolute():
        return split_path.resolve()

    return (
        dataset_root / split_path
    ).resolve()


def _label_directory_from_image_directory(
    image_directory: Path,
) -> Path:
    """Convert an images split path into its corresponding labels path."""
    path_parts = list(image_directory.parts)

    image_directory_indexes = [
        index
        for index, part in enumerate(path_parts)
        if part == "images"
    ]

    if not image_directory_indexes:
        raise ValueError(
            "The image path does not contain an 'images' directory: "
            f"{image_directory}"
        )

    images_index = image_directory_indexes[-1]
    path_parts[images_index] = "labels"

    return Path(*path_parts)


def _expected_label_path(
    image_path: Path,
    image_directory: Path,
    label_directory: Path,
) -> Path:
    """Return the YOLO label path corresponding to an image path."""
    relative_image_path = image_path.relative_to(
        image_directory
    )

    return (
        label_directory / relative_image_path
    ).with_suffix(".txt")


def _collect_images(image_directory: Path) -> list[Path]:
    """Collect supported image files recursively from a split directory."""
    return sorted(
        path
        for path in image_directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in VALID_IMAGE_SUFFIXES
        )
    )


def _collect_labels(label_directory: Path) -> list[Path]:
    """Collect YOLO text-label files recursively from a split directory."""
    return sorted(
        path
        for path in label_directory.rglob("*.txt")
        if path.is_file()
    )


def _validate_label_file(
    label_path: Path,
    class_names: list[str],
) -> tuple[Counter[int], list[str]]:
    """Validate one YOLO detection label file and count its objects."""
    class_counts: Counter[int] = Counter()
    errors: list[str] = []

    try:
        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()
    except OSError as error:
        return (
            class_counts,
            [f"Could not read label file {label_path}: {error}"],
        )

    for line_number, line in enumerate(lines, start=1):
        stripped_line = line.strip()

        if not stripped_line:
            continue

        values = stripped_line.split()
        location = f"{label_path}:{line_number}"

        if len(values) != 5:
            errors.append(
                f"{location}: expected 5 values but found "
                f"{len(values)}"
            )
            continue

        try:
            class_value = float(values[0])
            center_x = float(values[1])
            center_y = float(values[2])
            width = float(values[3])
            height = float(values[4])
        except ValueError:
            errors.append(
                f"{location}: contains a nonnumeric value"
            )
            continue

        all_values = [
            class_value,
            center_x,
            center_y,
            width,
            height,
        ]

        if not all(
            math.isfinite(value)
            for value in all_values
        ):
            errors.append(
                f"{location}: all values must be finite numbers"
            )
            continue

        if not class_value.is_integer():
            errors.append(
                f"{location}: class ID must be an integer"
            )
            continue

        class_id = int(class_value)

        if not 0 <= class_id < len(class_names):
            errors.append(
                f"{location}: invalid class ID {class_id}; "
                f"expected a value from 0 through "
                f"{len(class_names) - 1}"
            )
            continue

        coordinate_errors_found = False

        if not 0.0 <= center_x <= 1.0:
            errors.append(
                f"{location}: center_x must be within [0, 1]; "
                f"found {center_x}"
            )
            coordinate_errors_found = True

        if not 0.0 <= center_y <= 1.0:
            errors.append(
                f"{location}: center_y must be within [0, 1]; "
                f"found {center_y}"
            )
            coordinate_errors_found = True

        if not 0.0 < width <= 1.0:
            errors.append(
                f"{location}: width must be within (0, 1]; "
                f"found {width}"
            )
            coordinate_errors_found = True

        if not 0.0 < height <= 1.0:
            errors.append(
                f"{location}: height must be within (0, 1]; "
                f"found {height}"
            )
            coordinate_errors_found = True

        if coordinate_errors_found:
            continue

        left = center_x - width / 2.0
        right = center_x + width / 2.0
        top = center_y - height / 2.0
        bottom = center_y + height / 2.0

        if (
            left < -BOX_TOLERANCE
            or top < -BOX_TOLERANCE
            or right > 1.0 + BOX_TOLERANCE
            or bottom > 1.0 + BOX_TOLERANCE
        ):
            errors.append(
                f"{location}: bounding box extends outside the image; "
                f"left={left:.6f}, "
                f"top={top:.6f}, "
                f"right={right:.6f}, "
                f"bottom={bottom:.6f}"
            )
            continue

        class_counts[class_id] += 1

    return class_counts, errors


def _validate_split(
    split_name: str,
    image_directory: Path,
    class_names: list[str],
) -> tuple[SplitSummary | None, list[str], list[str]]:
    """Validate all images and labels belonging to one dataset split."""
    errors: list[str] = []
    warnings: list[str] = []

    if not image_directory.is_dir():
        errors.append(
            f"Missing {split_name} image directory: "
            f"{image_directory}"
        )
        return None, errors, warnings

    try:
        label_directory = (
            _label_directory_from_image_directory(
                image_directory
            )
        )
    except ValueError as error:
        errors.append(str(error))
        return None, errors, warnings

    if not label_directory.is_dir():
        errors.append(
            f"Missing {split_name} label directory: "
            f"{label_directory}"
        )
        return None, errors, warnings

    images = _collect_images(image_directory)
    labels = _collect_labels(label_directory)

    if not images:
        errors.append(
            f"No supported images were found in: "
            f"{image_directory}"
        )

    expected_label_paths: set[Path] = set()
    class_counts_by_id: Counter[int] = Counter()
    empty_label_count = 0

    for image_path in images:
        label_path = _expected_label_path(
            image_path=image_path,
            image_directory=image_directory,
            label_directory=label_directory,
        )

        expected_label_paths.add(
            label_path.resolve()
        )

        if not label_path.is_file():
            errors.append(
                f"Missing label for image: {image_path}"
            )
            continue

        try:
            label_text = label_path.read_text(
                encoding="utf-8"
            )
        except OSError as error:
            errors.append(
                f"Could not read label file "
                f"{label_path}: {error}"
            )
            continue

        if not label_text.strip():
            empty_label_count += 1
            warnings.append(
                f"Empty label file treated as a background image: "
                f"{label_path}"
            )
            continue

        file_counts, file_errors = _validate_label_file(
            label_path=label_path,
            class_names=class_names,
        )

        class_counts_by_id.update(file_counts)
        errors.extend(file_errors)

    actual_label_paths = {
        label_path.resolve()
        for label_path in labels
    }

    orphan_label_paths = sorted(
        actual_label_paths - expected_label_paths
    )

    for orphan_label_path in orphan_label_paths:
        errors.append(
            "Label file has no matching image: "
            f"{orphan_label_path}"
        )

    class_counts = {
        class_name: class_counts_by_id.get(
            class_id,
            0,
        )
        for class_id, class_name in enumerate(
            class_names
        )
    }

    split_summary = SplitSummary(
        split=split_name,
        image_count=len(images),
        label_count=len(labels),
        empty_label_count=empty_label_count,
        object_count=sum(
            class_counts_by_id.values()
        ),
        class_counts=class_counts,
    )

    return split_summary, errors, warnings


def _load_import_summary(
    import_summary_path: Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Load import_summary.json and return parsing errors separately."""
    errors: list[str] = []

    if not import_summary_path.is_file():
        return None, errors

    try:
        summary = json.loads(
            import_summary_path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as error:
        errors.append(
            f"Invalid JSON in {import_summary_path}: {error}"
        )
        return None, errors
    except OSError as error:
        errors.append(
            f"Could not read {import_summary_path}: {error}"
        )
        return None, errors

    if not isinstance(summary, dict):
        errors.append(
            f"{import_summary_path} must contain a JSON object."
        )
        return None, errors

    return summary, errors


def _compare_import_summary(
    summary: dict[str, Any],
    class_names: list[str],
    split_summaries: list[SplitSummary],
) -> tuple[list[str], list[str]]:
    """Compare available import-summary metadata against dataset results."""
    errors: list[str] = []
    warnings: list[str] = []

    summary_class_names = summary.get("class_names")

    if isinstance(summary_class_names, list):
        normalized_summary_names = [
            str(class_name)
            for class_name in summary_class_names
        ]

        if normalized_summary_names != class_names:
            errors.append(
                "Class names in import_summary.json do not match "
                "dataset.yaml. "
                f"Summary: {normalized_summary_names}; "
                f"YAML: {class_names}"
            )

    summary_splits = summary.get("splits")

    if not isinstance(summary_splits, list):
        warnings.append(
            "import_summary.json does not contain a comparable "
            "'splits' list. The file was valid JSON, but detailed "
            "count comparison was skipped."
        )
        return errors, warnings

    indexed_splits = {
        str(split_data.get("split")): split_data
        for split_data in summary_splits
        if isinstance(split_data, dict)
    }

    for split_summary in split_summaries:
        imported_split = indexed_splits.get(
            split_summary.split
        )

        if imported_split is None:
            warnings.append(
                "import_summary.json does not contain an entry for "
                f"the '{split_summary.split}' split."
            )
            continue

        imported_image_count = imported_split.get(
            "images_exported"
        )

        if (
            imported_image_count is not None
            and int(imported_image_count)
            != split_summary.image_count
        ):
            errors.append(
                f"{split_summary.split}: image total does not match "
                "import_summary.json "
                f"({split_summary.image_count} != "
                f"{imported_image_count})"
            )

        imported_object_count = imported_split.get(
            "objects_exported"
        )

        if (
            imported_object_count is not None
            and int(imported_object_count)
            != split_summary.object_count
        ):
            errors.append(
                f"{split_summary.split}: object total does not match "
                "import_summary.json "
                f"({split_summary.object_count} != "
                f"{imported_object_count})"
            )

        imported_class_counts = imported_split.get(
            "class_counts"
        )

        if isinstance(imported_class_counts, dict):
            normalized_counts = {
                str(class_name): int(count)
                for class_name, count
                in imported_class_counts.items()
            }

            if normalized_counts != split_summary.class_counts:
                errors.append(
                    f"{split_summary.split}: per-class counts do not "
                    "match import_summary.json. "
                    f"Validator: {split_summary.class_counts}; "
                    f"Summary: {normalized_counts}"
                )

    return errors, warnings


def validate_yolo_dataset(
    dataset_yaml: Path,
    expected_class_names: list[str] | None = None,
    import_summary_path: Path | None = None,
) -> ValidationReport:
    """Validate YOLO paths, class mappings, labels, boxes, and metadata."""
    dataset_yaml = dataset_yaml.expanduser().resolve()

    config = _load_yaml_mapping(
        dataset_yaml
    )

    dataset_root = _resolve_dataset_root(
        dataset_yaml=dataset_yaml,
        config=config,
    )

    errors: list[str] = []
    warnings: list[str] = []
    split_summaries: list[SplitSummary] = []

    if not dataset_root.is_dir():
        errors.append(
            "The dataset root does not exist: "
            f"{dataset_root}. If the project directory was moved, "
            "rerun scripts/import_lars.py to regenerate dataset.yaml."
        )

    class_names = _normalize_class_names(
        config.get("names")
    )

    if (
        expected_class_names is not None
        and class_names != expected_class_names
    ):
        errors.append(
            "The dataset class order does not match the project "
            "configuration. "
            f"Expected: {expected_class_names}; "
            f"found: {class_names}"
        )

    configured_class_count = config.get("nc")

    if (
        configured_class_count is not None
        and int(configured_class_count)
        != len(class_names)
    ):
        errors.append(
            "The 'nc' value in dataset.yaml does not match the "
            f"number of class names "
            f"({configured_class_count} != {len(class_names)})."
        )

    for split_name in ("train", "val"):
        split_value = config.get(split_name)

        if split_value is None:
            errors.append(
                f"dataset.yaml is missing the '{split_name}' entry."
            )
            continue

        image_directory = _resolve_split_path(
            dataset_root=dataset_root,
            split_value=split_value,
        )

        split_summary, split_errors, split_warnings = (
            _validate_split(
                split_name=split_name,
                image_directory=image_directory,
                class_names=class_names,
            )
        )

        if split_summary is not None:
            split_summaries.append(
                split_summary
            )

        errors.extend(split_errors)
        warnings.extend(split_warnings)

    resolved_summary_path = (
        import_summary_path.expanduser().resolve()
        if import_summary_path is not None
        else dataset_root / "import_summary.json"
    )

    if not resolved_summary_path.is_file():
        warnings.append(
            f"Import summary not found: {resolved_summary_path}"
        )
    else:
        import_summary, summary_errors = (
            _load_import_summary(
                resolved_summary_path
            )
        )

        errors.extend(summary_errors)

        if import_summary is not None:
            comparison_errors, comparison_warnings = (
                _compare_import_summary(
                    summary=import_summary,
                    class_names=class_names,
                    split_summaries=split_summaries,
                )
            )

            errors.extend(comparison_errors)
            warnings.extend(comparison_warnings)

    return ValidationReport(
        dataset_yaml=str(dataset_yaml),
        dataset_root=str(dataset_root),
        class_names=class_names,
        splits=split_summaries,
        errors=errors,
        warnings=warnings,
    )