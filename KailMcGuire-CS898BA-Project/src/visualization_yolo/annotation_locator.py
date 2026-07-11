"""
Utilities for locating matching YOLO image and annotation files.

This module contains Stage 2 of the annotation visualization pipeline. It
finds an image with a corresponding non-empty YOLO label file and reads the
raw annotation rows.

It does not parse coordinate values or open image pixels.
"""

from dataclasses import dataclass
from pathlib import Path

from .dataset_verifier import SUPPORTED_IMAGE_EXTENSIONS


@dataclass(frozen=True)
class AnnotationPairResult:
    """
    Store one matched image and YOLO label file.

    Attributes
    ----------
    image_path:
        Path to the selected image.

    label_path:
        Path to the matching YOLO text file.

    annotation_lines:
        Non-empty raw annotation rows read from the label file.

    label_directory:
        Directory containing the split's YOLO label files.
    """

    image_path: Path
    label_path: Path
    annotation_lines: list[str]
    label_directory: Path


class YoloAnnotationLocator:
    """
    Locate an annotated image-label pair within one YOLO dataset split.

    YOLO pairs images and labels using the same filename stem:

        images/train/example.jpg
        labels/train/example.txt

    This class searches image files in deterministic alphabetical order and
    returns the first image whose matching label file contains at least one
    annotation.

    Parameters
    ----------
    dataset_root:
        Absolute root path of the converted YOLO dataset.

    image_directory:
        Image directory for the selected dataset split.

    split_name:
        Split whose labels should be searched, such as ``train`` or ``val``.
    """

    def __init__(
        self,
        dataset_root: Path,
        image_directory: Path,
        split_name: str,
    ) -> None:
        """
        Initialize the annotation locator.

        Parameters
        ----------
        dataset_root:
            Root directory containing the ``images`` and ``labels`` folders.

        image_directory:
            Directory containing images for the selected split.

        split_name:
            Split name used to locate ``labels/<split_name>``.

        Raises
        ------
        ValueError
            Raised when the split name is empty.
        """
        if not isinstance(split_name, str) or not split_name.strip():
            raise ValueError(
                "split_name must be a non-empty string."
            )

        self.dataset_root = dataset_root.resolve()
        self.image_directory = image_directory.resolve()
        self.split_name = split_name.strip()

    def resolve_label_directory(self) -> Path:
        """
        Resolve and validate the split's YOLO label directory.

        Returns
        -------
        Path
            Absolute path to ``labels/<split_name>``.

        Raises
        ------
        FileNotFoundError
            Raised when the expected directory does not exist.

        NotADirectoryError
            Raised when the resolved path is not a directory.
        """
        label_directory = (
            self.dataset_root
            / "labels"
            / self.split_name
        ).resolve()

        if not label_directory.exists():
            raise FileNotFoundError(
                "The expected YOLO label directory does not exist:\n"
                f"{label_directory}"
            )

        if not label_directory.is_dir():
            raise NotADirectoryError(
                "The YOLO label path is not a directory:\n"
                f"{label_directory}"
            )

        return label_directory

    @staticmethod
    def read_label_lines(label_path: Path) -> list[str]:
        """
        Read all non-empty lines from one YOLO annotation file.

        A standard YOLO row follows this format:

            class_id center_x center_y width height

        This method deliberately leaves the values as strings. Numeric parsing
        and validation belong to Stage 3.

        Parameters
        ----------
        label_path:
            Path to a YOLO ``.txt`` annotation file.

        Returns
        -------
        list[str]
            Non-empty annotation rows.

        Raises
        ------
        FileNotFoundError
            Raised when the annotation file does not exist.

        ValueError
            Raised when the file does not use the ``.txt`` extension.
        """
        if not label_path.exists():
            raise FileNotFoundError(
                f"The YOLO label file does not exist:\n{label_path}"
            )

        if label_path.suffix.lower() != ".txt":
            raise ValueError(
                "YOLO annotation files must use the .txt extension."
            )

        with label_path.open("r", encoding="utf-8") as label_file:
            return [
                line.strip()
                for line in label_file
                if line.strip()
            ]

    def find_annotated_pair(self) -> AnnotationPairResult:
        """
        Find the first image with a matching non-empty label file.

        Images are sorted alphabetically so repeated runs select the same
        sample unless the dataset contents change.

        Images with missing labels or empty labels are skipped because they
        contain no bounding boxes to visualize.

        Returns
        -------
        AnnotationPairResult
            Selected image, matching label, raw annotations, and label
            directory.

        Raises
        ------
        FileNotFoundError
            Raised when no annotated image-label pair can be found.
        """
        label_directory = self.resolve_label_directory()

        image_paths = sorted(
            file_path
            for file_path in self.image_directory.iterdir()
            if file_path.is_file()
            and file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )

        for image_path in image_paths:
            label_path = (
                label_directory
                / f"{image_path.stem}.txt"
            )

            if not label_path.exists():
                continue

            annotation_lines = self.read_label_lines(
                label_path
            )

            if annotation_lines:
                return AnnotationPairResult(
                    image_path=image_path,
                    label_path=label_path,
                    annotation_lines=annotation_lines,
                    label_directory=label_directory,
                )

        raise FileNotFoundError(
            "No annotated image-label pair could be found.\n"
            f"Images: {self.image_directory}\n"
            f"Labels: {label_directory}"
        )