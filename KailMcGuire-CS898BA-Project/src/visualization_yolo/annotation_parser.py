"""
Parsing and coordinate-conversion tools for YOLO annotations.

This module contains Stage 3 of the visualization pipeline. It converts raw
YOLO annotation rows into structured Python objects and converts normalized
YOLO boxes into image pixel coordinates.

This module does not draw the boxes. Drawing will be added in Stage 4.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2 as cv


@dataclass(frozen=True)
class YoloAnnotation:
    """
    Represent one parsed YOLO object-detection annotation.

    Attributes
    ----------
    class_id:
        Numeric category identifier stored in the label file.

    class_name:
        Readable class name obtained from ``dataset.yaml``.

    center_x:
        Horizontal box center normalized by image width.

    center_y:
        Vertical box center normalized by image height.

    width:
        Box width normalized by image width.

    height:
        Box height normalized by image height.
    """

    class_id: int
    class_name: str
    center_x: float
    center_y: float
    width: float
    height: float


class YoloAnnotationParser:
    """
    Parse YOLO annotation rows and convert them into pixel coordinates.

    Parameters
    ----------
    class_names:
        Mapping from numeric class IDs to readable category names.
    """

    def __init__(
        self,
        class_names: dict[int, str],
    ) -> None:
        """
        Initialize the parser with the configured dataset classes.

        Parameters
        ----------
        class_names:
            Dictionary such as ``{0: "vessel", 1: "buoy"}``.

        Raises
        ------
        ValueError
            Raised when the class-name mapping is empty.
        """
        if not class_names:
            raise ValueError(
                "At least one configured class name is required."
            )

        self.class_names = class_names

    def parse_line(
        self,
        annotation_line: str,
    ) -> YoloAnnotation:
        """
        Parse and validate one YOLO annotation row.

        A valid row contains exactly five whitespace-separated values:

            class_id center_x center_y width height

        The class ID must exist in the configured class mapping. Center
        coordinates must fall between 0 and 1. Width and height must be
        greater than 0 and no greater than 1.

        Parameters
        ----------
        annotation_line:
            One raw annotation row from a YOLO label file.

        Returns
        -------
        YoloAnnotation
            Structured, validated annotation.

        Raises
        ------
        ValueError
            Raised when the row has the wrong number of values, contains
            invalid numeric data, references an unknown class, or contains
            coordinates outside the accepted ranges.
        """
        values = annotation_line.split()

        if len(values) != 5:
            raise ValueError(
                "A YOLO annotation must contain exactly five values:\n"
                "class_id center_x center_y width height\n"
                f"Received: {annotation_line!r}"
            )

        (
            class_id_text,
            center_x_text,
            center_y_text,
            width_text,
            height_text,
        ) = values

        try:
            class_id = int(class_id_text)
        except ValueError as error:
            raise ValueError(
                f"The class ID must be an integer: {class_id_text!r}"
            ) from error

        if class_id not in self.class_names:
            raise ValueError(
                f"Class ID {class_id} is not defined in dataset.yaml."
            )

        try:
            center_x = float(center_x_text)
            center_y = float(center_y_text)
            width = float(width_text)
            height = float(height_text)
        except ValueError as error:
            raise ValueError(
                "YOLO bounding-box coordinates must be numeric.\n"
                f"Received: {annotation_line!r}"
            ) from error

        if not 0.0 <= center_x <= 1.0:
            raise ValueError(
                f"center_x must be between 0 and 1: {center_x}"
            )

        if not 0.0 <= center_y <= 1.0:
            raise ValueError(
                f"center_y must be between 0 and 1: {center_y}"
            )

        if not 0.0 < width <= 1.0:
            raise ValueError(
                f"width must be greater than 0 and at most 1: {width}"
            )

        if not 0.0 < height <= 1.0:
            raise ValueError(
                f"height must be greater than 0 and at most 1: {height}"
            )

        return YoloAnnotation(
            class_id=class_id,
            class_name=self.class_names[class_id],
            center_x=center_x,
            center_y=center_y,
            width=width,
            height=height,
        )

    def parse_lines(
        self,
        annotation_lines: list[str],
    ) -> list[YoloAnnotation]:
        """
        Parse all annotation rows from one YOLO label file.

        Parameters
        ----------
        annotation_lines:
            Raw, non-empty annotation rows.

        Returns
        -------
        list[YoloAnnotation]
            Parsed annotations in their original file order.

        Raises
        ------
        ValueError
            Propagated when any individual annotation row is invalid.
        """
        return [
            self.parse_line(annotation_line)
            for annotation_line in annotation_lines
        ]

    @staticmethod
    def load_image_dimensions(
        image_path: Path,
    ) -> tuple[int, int]:
        """
        Decode an image and return its width and height.

        OpenCV stores an image shape as:

            height, width, channels

        This method returns ``width, height`` because that ordering corresponds
        more directly to bounding-box coordinate calculations.

        Parameters
        ----------
        image_path:
            Path to the image that will eventually be visualized.

        Returns
        -------
        tuple[int, int]
            Image width and height in pixels.

        Raises
        ------
        FileNotFoundError
            Raised when the image does not exist.

        ValueError
            Raised when OpenCV cannot decode the image or when its dimensions
            are invalid.
        """
        if not image_path.exists():
            raise FileNotFoundError(
                f"The image file does not exist:\n{image_path}"
            )

        image = cv.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"OpenCV could not decode the image:\n{image_path}"
            )

        image_height, image_width = image.shape[:2]

        if image_width <= 0 or image_height <= 0:
            raise ValueError(
                "The image has invalid dimensions:\n"
                f"Width: {image_width}, Height: {image_height}"
            )

        return image_width, image_height

    @staticmethod
    def convert_to_pixel_box(
        annotation: YoloAnnotation,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int]:
        """
        Convert a normalized YOLO box into pixel corner coordinates.

        YOLO stores boxes using:

            center_x, center_y, width, height

        OpenCV drawing functions require:

            left, top, right, bottom

        The normalized values are first multiplied by the image dimensions.
        Half the calculated width and height are then subtracted from or added
        to the center point to obtain the four edges.

        The resulting values are clamped to valid image boundaries.

        Parameters
        ----------
        annotation:
            Parsed YOLO annotation.

        image_width:
            Full image width in pixels.

        image_height:
            Full image height in pixels.

        Returns
        -------
        tuple[int, int, int, int]
            ``left, top, right, bottom`` pixel coordinates.

        Raises
        ------
        ValueError
            Raised when the image dimensions are invalid or when rounding and
            clamping produce a zero-area pixel box.
        """
        if image_width <= 0 or image_height <= 0:
            raise ValueError(
                "Image width and height must both be positive."
            )

        center_x_pixels = annotation.center_x * image_width
        center_y_pixels = annotation.center_y * image_height
        box_width_pixels = annotation.width * image_width
        box_height_pixels = annotation.height * image_height

        left = round(
            center_x_pixels - box_width_pixels / 2
        )

        right = round(
            center_x_pixels + box_width_pixels / 2
        )

        top = round(
            center_y_pixels - box_height_pixels / 2
        )

        bottom = round(
            center_y_pixels + box_height_pixels / 2
        )

        left = max(0, min(left, image_width - 1))
        right = max(0, min(right, image_width - 1))
        top = max(0, min(top, image_height - 1))
        bottom = max(0, min(bottom, image_height - 1))

        if right <= left or bottom <= top:
            raise ValueError(
                "The converted box has zero or negative pixel area.\n"
                f"Class: {annotation.class_name}\n"
                f"Box: ({left}, {top}, {right}, {bottom})"
            )

        return left, top, right, bottom