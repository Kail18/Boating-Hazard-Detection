"""Image-processing utilities for maritime imagery."""

from pathlib import Path

import cv2 as cv
import numpy as np


def apply_clahe_bgr(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Improve local contrast by applying CLAHE to the LAB luminance channel."""
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    luminance, channel_a, channel_b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced_luminance = clahe.apply(luminance)
    enhanced_lab = cv.merge((enhanced_luminance, channel_a, channel_b))
    return cv.cvtColor(enhanced_lab, cv.COLOR_LAB2BGR)


def resize_with_padding(
    image: np.ndarray,
    target_size: tuple[int, int] = (640, 640),
) -> np.ndarray:
    """Resize an image without stretching it, then pad to the target size."""
    target_width, target_height = target_size
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv.resize(image, (resized_width, resized_height))

    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - resized_width) // 2
    y_offset = (target_height - resized_height) // 2
    canvas[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized
    return canvas


def preprocess_file(input_path: Path, output_path: Path) -> None:
    """Read, enhance, resize, and save one image."""
    image = cv.imread(str(input_path))
    if image is None:
        raise ValueError(f"Could not read image: {input_path}")

    processed = resize_with_padding(apply_clahe_bgr(image))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(output_path), processed):
        raise OSError(f"Could not save image: {output_path}")
