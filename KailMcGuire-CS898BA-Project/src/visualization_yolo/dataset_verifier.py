"""
Utilities for validating a YOLO-format dataset before visualization.

This module contains the Stage 1 responsibilities of the annotation
visualization pipeline. It validates the dataset configuration, resolves
training and validation directories, normalizes class names, and confirms
that the expected image files are available.

No images are decoded and no annotation files are read in this module.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


@dataclass(frozen=True)
class DatasetVerificationResult:
    """
    Store the validated information needed by later visualization stages.

    This object collects the results of Stage 1 so that later classes do not
    need to reopen the dataset YAML or recalculate the same paths.

    Attributes
    ----------
    configuration:
        Complete dictionary loaded from the YOLO dataset YAML file.

    dataset_root:
        Absolute path to the root of the converted YOLO dataset.

    train_directory:
        Absolute path to the training image directory.

    validation_directory:
        Absolute path to the validation image directory.

    class_names:
        Mapping from numeric YOLO class IDs to readable class names.

    train_image_count:
        Number of supported image files in the training directory.

    validation_image_count:
        Number of supported image files in the validation directory.
    """

    configuration: dict[str, Any]
    dataset_root: Path
    train_directory: Path
    validation_directory: Path
    class_names: dict[int, str]
    train_image_count: int
    validation_image_count: int


class YoloDatasetVerifier:
    """
    Validate the configuration and image directories of a YOLO dataset.

    This class performs the first stage of the annotation visualization
    workflow. It confirms that the generated dataset configuration can be
    loaded and that its paths point to valid image directories.

    The verifier does not read YOLO label files and does not draw bounding
    boxes. Its purpose is to establish a reliable dataset foundation before
    later visualization operations begin.

    Parameters
    ----------
    dataset_yaml:
        Path to the YOLO ``dataset.yaml`` file generated during dataset
        conversion.
    """

    def __init__(self, dataset_yaml: Path) -> None:
        """
        Initialize the verifier with the dataset configuration path.

        Parameters
        ----------
        dataset_yaml:
            Path to the YOLO dataset YAML file.

        Raises
        ------
        TypeError
            Raised when ``dataset_yaml`` is not a ``Path`` object.
        """
        if not isinstance(dataset_yaml, Path):
            raise TypeError(
                "dataset_yaml must be provided as a pathlib.Path object."
            )

        self.dataset_yaml = dataset_yaml.resolve()

    def load_configuration(self) -> dict[str, Any]:
        """
        Load and validate the top-level YOLO dataset configuration.

        A YOLO dataset YAML file normally contains:

            path: /path/to/dataset
            train: images/train
            val: images/val
            names:
              0: vessel
              1: buoy

        This method confirms that:

        - The YAML file exists.
        - The file contains valid YAML.
        - The top-level value is a dictionary.
        - The ``train``, ``val``, and ``names`` fields are present.

        Returns
        -------
        dict[str, Any]
            Parsed dataset configuration.

        Raises
        ------
        FileNotFoundError
            Raised when the YAML file cannot be found.

        ValueError
            Raised when the file is empty, malformed, or missing a required
            field.
        """
        if not self.dataset_yaml.exists():
            raise FileNotFoundError(
                "The YOLO dataset configuration could not be found:\n"
                f"{self.dataset_yaml}"
            )

        with self.dataset_yaml.open("r", encoding="utf-8") as yaml_file:
            configuration = yaml.safe_load(yaml_file)

        if configuration is None:
            raise ValueError(
                f"The dataset configuration is empty: {self.dataset_yaml}"
            )

        if not isinstance(configuration, dict):
            raise ValueError(
                "The dataset YAML must contain a dictionary at its top level."
            )

        required_fields = {"train", "val", "names"}
        missing_fields = required_fields - configuration.keys()

        if missing_fields:
            missing_text = ", ".join(sorted(missing_fields))

            raise ValueError(
                "The dataset configuration is missing required fields: "
                f"{missing_text}"
            )

        return configuration

    @staticmethod
    def normalize_class_names(
        names_value: list[str] | dict[int | str, str],
    ) -> dict[int, str]:
        """
        Convert the YAML class-name field into a consistent dictionary.

        Ultralytics supports class names as either a list:

            names:
              - vessel
              - buoy

        or a dictionary:

            names:
              0: vessel
              1: buoy

        Later annotation code needs a dependable mapping from integer IDs to
        class names. This method converts either format into:

            {
                0: "vessel",
                1: "buoy"
            }

        Parameters
        ----------
        names_value:
            Value stored under the YAML ``names`` field.

        Returns
        -------
        dict[int, str]
            Sorted mapping of integer class IDs to class names.

        Raises
        ------
        ValueError
            Raised when the names field has an unsupported structure or
            contains invalid class identifiers or names.
        """
        if isinstance(names_value, list):
            class_names = {
                class_id: class_name
                for class_id, class_name in enumerate(names_value)
            }

        elif isinstance(names_value, dict):
            try:
                class_names = {
                    int(class_id): class_name
                    for class_id, class_name in names_value.items()
                }
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Every class ID must be convertible to an integer."
                ) from error

        else:
            raise ValueError(
                "The YAML names field must be a list or dictionary."
            )

        if not class_names:
            raise ValueError(
                "The dataset YAML does not contain any class names."
            )

        for class_id, class_name in class_names.items():
            if not isinstance(class_name, str) or not class_name.strip():
                raise ValueError(
                    f"Class ID {class_id} has an invalid class name."
                )

        return dict(sorted(class_names.items()))

    def resolve_dataset_root(
        self,
        configuration: dict[str, Any],
    ) -> Path:
        """
        Resolve the absolute root directory of the YOLO dataset.

        The YAML may contain an absolute or relative ``path`` field. When the
        field is absent, the directory containing ``dataset.yaml`` is used.

        Relative dataset paths are interpreted relative to the directory
        containing the YAML file.

        Parameters
        ----------
        configuration:
            Parsed YOLO dataset configuration.

        Returns
        -------
        Path
            Absolute dataset root path.
        """
        configured_root = configuration.get("path")

        if configured_root is None:
            return self.dataset_yaml.parent.resolve()

        root_path = Path(configured_root).expanduser()

        if not root_path.is_absolute():
            root_path = self.dataset_yaml.parent / root_path

        return root_path.resolve()

    @staticmethod
    def resolve_split_directory(
        dataset_root: Path,
        split_value: str,
    ) -> Path:
        """
        Resolve a training or validation YAML path into an absolute path.

        Parameters
        ----------
        dataset_root:
            Absolute root directory of the YOLO dataset.

        split_value:
            Value stored under the YAML ``train`` or ``val`` field.

        Returns
        -------
        Path
            Absolute image-directory path.

        Raises
        ------
        ValueError
            Raised when the split entry is not a non-empty string.
        """
        if not isinstance(split_value, str) or not split_value.strip():
            raise ValueError(
                "Dataset split paths must be non-empty strings."
            )

        split_path = Path(split_value).expanduser()

        if not split_path.is_absolute():
            split_path = dataset_root / split_path

        return split_path.resolve()

    @staticmethod
    def validate_image_directory(
        split_name: str,
        image_directory: Path,
    ) -> int:
        """
        Confirm that a split directory exists and contains supported images.

        This method checks filenames and extensions only. It does not decode
        every image, which keeps Stage 1 fast even when the dataset contains
        thousands of files.

        Parameters
        ----------
        split_name:
            Readable split name used in error messages.

        image_directory:
            Directory expected to contain the split's images.

        Returns
        -------
        int
            Number of supported image files found.

        Raises
        ------
        FileNotFoundError
            Raised when the directory does not exist.

        NotADirectoryError
            Raised when the path exists but is not a directory.

        ValueError
            Raised when no supported image files are found.
        """
        if not image_directory.exists():
            raise FileNotFoundError(
                f"The {split_name} image directory does not exist:\n"
                f"{image_directory}"
            )

        if not image_directory.is_dir():
            raise NotADirectoryError(
                f"The {split_name} image path is not a directory:\n"
                f"{image_directory}"
            )

        image_count = sum(
            1
            for file_path in image_directory.iterdir()
            if file_path.is_file()
            and file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )

        if image_count == 0:
            raise ValueError(
                f"No supported images were found in:\n{image_directory}"
            )

        return image_count

    def verify(self) -> DatasetVerificationResult:
        """
        Run the complete Stage 1 dataset-verification process.

        The method loads the YAML file, normalizes class names, resolves image
        directories, counts the images, and returns all verified information
        in one immutable result object.

        Returns
        -------
        DatasetVerificationResult
            Validated configuration and directory information for later
            visualization stages.
        """
        configuration = self.load_configuration()

        class_names = self.normalize_class_names(
            configuration["names"]
        )

        dataset_root = self.resolve_dataset_root(
            configuration
        )

        train_directory = self.resolve_split_directory(
            dataset_root=dataset_root,
            split_value=configuration["train"],
        )

        validation_directory = self.resolve_split_directory(
            dataset_root=dataset_root,
            split_value=configuration["val"],
        )

        train_image_count = self.validate_image_directory(
            split_name="training",
            image_directory=train_directory,
        )

        validation_image_count = self.validate_image_directory(
            split_name="validation",
            image_directory=validation_directory,
        )

        return DatasetVerificationResult(
            configuration=configuration,
            dataset_root=dataset_root,
            train_directory=train_directory,
            validation_directory=validation_directory,
            class_names=class_names,
            train_image_count=train_image_count,
            validation_image_count=validation_image_count,
        )