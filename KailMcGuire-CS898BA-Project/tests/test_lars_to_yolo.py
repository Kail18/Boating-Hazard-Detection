"""Tests for LaRS panoptic-to-YOLO conversion helpers."""

from pathlib import Path

import pytest

from src.data.lars_to_yolo import canonicalize_category, yolo_bbox


def test_canonicalize_category_handles_common_separators() -> None:
    assert canonicalize_category("Boat/Ship") == "boat ship"
    assert canonicalize_category("paddle_board") == "paddle board"
    assert canonicalize_category("Row-Boat") == "row boat"


def test_yolo_bbox_converts_coco_box() -> None:
    center_x, center_y, width, height = yolo_bbox([10, 20, 40, 20], 100, 100)
    assert center_x == pytest.approx(0.30)
    assert center_y == pytest.approx(0.30)
    assert width == pytest.approx(0.40)
    assert height == pytest.approx(0.20)


def test_yolo_bbox_clamps_to_image() -> None:
    center_x, center_y, width, height = yolo_bbox([-10, 90, 30, 20], 100, 100)
    assert center_x == pytest.approx(0.10)
    assert center_y == pytest.approx(0.95)
    assert width == pytest.approx(0.20)
    assert height == pytest.approx(0.10)
