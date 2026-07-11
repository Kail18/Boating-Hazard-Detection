import numpy as np

from src.processing.preprocess import apply_clahe_bgr, resize_with_padding


def test_resize_with_padding_shape() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    result = resize_with_padding(image, (640, 640))
    assert result.shape == (640, 640, 3)


def test_clahe_preserves_shape() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    result = apply_clahe_bgr(image)
    assert result.shape == image.shape
