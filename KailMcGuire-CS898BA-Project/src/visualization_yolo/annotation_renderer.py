"""
Drawing and image-output utilities for YOLO annotation visualization.

This module contains Stage 4 of the visualization pipeline. It loads one
source image, draws the parsed ground-truth bounding boxes and class names,
and saves the annotated image to an output directory.

This module is intentionally limited to rendering annotations that have
already been located, parsed, and validated by the earlier visualization
stages. It does not search for images, parse label-file text, or modify the
original dataset images.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from .annotation_parser import (
    YoloAnnotation,
    YoloAnnotationParser,
)


@dataclass(frozen=True)
class RenderedImageResult:
    """
    Store information about one successfully rendered annotation image.

    Attributes
    ----------
    source_image_path:
        Path to the original unmodified dataset image.

    output_image_path:
        Path where the annotated visualization was saved.

    object_count:
        Number of bounding boxes drawn on the image.

    image_width:
        Width of the rendered image in pixels.

    image_height:
        Height of the rendered image in pixels.
    """

    source_image_path: Path
    output_image_path: Path
    object_count: int
    image_width: int
    image_height: int


class YoloAnnotationRenderer:
    """
    Draw parsed YOLO annotations on an image and save the visualization.

    The renderer receives annotations that have already been validated by
    :class:`YoloAnnotationParser`. It converts each normalized annotation
    into pixel coordinates, draws a bounding-box rectangle, and places a
    readable class label near the box.

    The original dataset image is never modified. OpenCV loads the image into
    memory, the renderer draws on a copy of that image array, and the completed
    visualization is written to a separate output directory.

    Parameters
    ----------
    output_directory:
        Directory where annotated image visualizations should be saved.

    box_thickness:
        Thickness of bounding-box outlines in pixels.

    font_scale:
        OpenCV text scaling value used for class labels.

    text_thickness:
        Thickness of the class-label text in pixels.
    """

    def __init__(
        self,
        output_directory: Path,
        box_thickness: int = 3,
        font_scale: float = 0.7,
        text_thickness: int = 2,
    ) -> None:
        """
        Initialize the renderer and prepare its output directory.

        Parameters
        ----------
        output_directory:
            Destination directory for rendered images. The directory is
            created automatically when it does not already exist.

        box_thickness:
            Positive integer controlling the thickness of rectangle borders.

        font_scale:
            Positive OpenCV font scale for the annotation text.

        text_thickness:
            Positive integer controlling the thickness of label text.

        Raises
        ------
        ValueError
            Raised when a drawing parameter is zero or negative.
        """
        if box_thickness <= 0:
            raise ValueError(
                "box_thickness must be greater than zero."
            )

        if font_scale <= 0:
            raise ValueError(
                "font_scale must be greater than zero."
            )

        if text_thickness <= 0:
            raise ValueError(
                "text_thickness must be greater than zero."
            )

        self.output_directory = output_directory.resolve()
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.text_thickness = text_thickness

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def load_image(
        image_path: Path,
    ) -> NDArray[np.uint8]:
        """
        Load an image into memory for annotation rendering.

        OpenCV reads color images using Blue-Green-Red channel order. This is
        appropriate for OpenCV drawing and saving functions, so no color-space
        conversion is required during the current stage.

        Parameters
        ----------
        image_path:
            Path to the original dataset image.

        Returns
        -------
        numpy.ndarray
            Decoded OpenCV image array.

        Raises
        ------
        FileNotFoundError
            Raised when the image path does not exist.

        ValueError
            Raised when OpenCV cannot decode the supplied image.
        """
        if not image_path.exists():
            raise FileNotFoundError(
                f"The source image does not exist:\n{image_path}"
            )

        image = cv.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"OpenCV could not decode the image:\n{image_path}"
            )

        return image

    @staticmethod
    def class_color(
        class_id: int,
    ) -> tuple[int, int, int]:
        """
        Return a consistent OpenCV BGR color for one class ID.

        Consistent class colors make multi-object visualizations easier to
        inspect. The same class receives the same color every time the script
        runs.

        OpenCV expects colors in Blue-Green-Red order rather than the more
        common Red-Green-Blue order.

        Parameters
        ----------
        class_id:
            Numeric class identifier from the YOLO annotation.

        Returns
        -------
        tuple[int, int, int]
            Three-element BGR color tuple.

        Notes
        -----
        The palette currently contains one color for each of the five project
        classes. The modulo operation allows the method to remain usable when
        additional class IDs are introduced later.
        """
        palette = [
            (255, 120, 0),
            (0, 200, 255),
            (0, 0, 255),
            (255, 0, 255),
            (0, 200, 0),
        ]

        return palette[class_id % len(palette)]

    def draw_label(
        self,
        image: NDArray[np.uint8],
        label_text: str,
        left: int,
        top: int,
        color: tuple[int, int, int],
    ) -> None:
        """
        Draw a readable class-name label near a bounding box.

        The method first measures the text dimensions using OpenCV. It then
        draws a filled rectangle behind the text so the class name remains
        visible against water, sky, glare, or dark objects.

        The preferred label position is directly above the bounding box. When
        insufficient space exists above the box, the label is moved inside the
        upper portion of the box.

        Parameters
        ----------
        image:
            Mutable OpenCV image array on which the label should be drawn.

        label_text:
            Text displayed near the bounding box, normally including both the
            class ID and class name.

        left:
            Left pixel coordinate of the corresponding bounding box.

        top:
            Top pixel coordinate of the corresponding bounding box.

        color:
            BGR color used for both the box and label background.

        Returns
        -------
        None
            The supplied image array is modified in place.
        """
        font = cv.FONT_HERSHEY_SIMPLEX

        (text_width, text_height), baseline = cv.getTextSize(
            label_text,
            font,
            self.font_scale,
            self.text_thickness,
        )

        padding = 5

        label_left = max(0, left)
        label_bottom = top

        if label_bottom - text_height - baseline - padding * 2 < 0:
            label_bottom = (
                top
                + text_height
                + baseline
                + padding * 2
            )

        image_height, image_width = image.shape[:2]

        label_right = min(
            image_width - 1,
            label_left + text_width + padding * 2,
        )

        label_top = max(
            0,
            label_bottom
            - text_height
            - baseline
            - padding * 2,
        )

        label_bottom = min(
            image_height - 1,
            label_bottom,
        )

        cv.rectangle(
            image,
            (label_left, label_top),
            (label_right, label_bottom),
            color,
            thickness=-1,
        )

        text_x = label_left + padding
        text_y = label_bottom - baseline - padding

        cv.putText(
            image,
            label_text,
            (text_x, text_y),
            font,
            self.font_scale,
            (255, 255, 255),
            self.text_thickness,
            lineType=cv.LINE_AA,
        )

    def render_annotations(
        self,
        image_path: Path,
        annotations: list[YoloAnnotation],
        annotation_parser: YoloAnnotationParser,
    ) -> RenderedImageResult:
        """
        Draw all supplied ground-truth annotations and save the result.

        The method performs the following steps:

        1. Load the original image.
        2. Create an independent copy for drawing.
        3. Convert every normalized YOLO box into pixel coordinates.
        4. Draw one rectangle per object.
        5. Draw the numeric class ID and readable class name.
        6. Save the completed image in the configured output directory.

        The output filename preserves the original image stem and adds the
        suffix ``_annotated``. For example:

            davimar_seq_01_00017.jpg

        becomes:

            davimar_seq_01_00017_annotated.jpg

        Parameters
        ----------
        image_path:
            Path to the original dataset image.

        annotations:
            Parsed and validated annotations belonging to the image.

        annotation_parser:
            Parser instance used to convert normalized annotations into pixel
            coordinates.

        Returns
        -------
        RenderedImageResult
            Details about the saved visualization.

        Raises
        ------
        ValueError
            Raised when no annotations are supplied or when OpenCV cannot save
            the completed visualization.
        """
        if not annotations:
            raise ValueError(
                "At least one annotation is required for rendering."
            )

        source_image = self.load_image(image_path)

        # Drawing on a copy guarantees that the source image array remains
        # unchanged in memory.
        rendered_image = source_image.copy()

        image_height, image_width = rendered_image.shape[:2]

        for annotation in annotations:
            left, top, right, bottom = (
                annotation_parser.convert_to_pixel_box(
                    annotation=annotation,
                    image_width=image_width,
                    image_height=image_height,
                )
            )

            color = self.class_color(
                annotation.class_id
            )

            cv.rectangle(
                rendered_image,
                (left, top),
                (right, bottom),
                color,
                thickness=self.box_thickness,
            )

            label_text = (
                f"{annotation.class_id}: "
                f"{annotation.class_name}"
            )

            self.draw_label(
                image=rendered_image,
                label_text=label_text,
                left=left,
                top=top,
                color=color,
            )

        output_image_path = (
            self.output_directory
            / f"{image_path.stem}_annotated{image_path.suffix}"
        )

        save_successful = cv.imwrite(
            str(output_image_path),
            rendered_image,
        )

        if not save_successful:
            raise ValueError(
                "OpenCV could not save the rendered image:\n"
                f"{output_image_path}"
            )

        return RenderedImageResult(
            source_image_path=image_path,
            output_image_path=output_image_path,
            object_count=len(annotations),
            image_width=image_width,
            image_height=image_height,
        )