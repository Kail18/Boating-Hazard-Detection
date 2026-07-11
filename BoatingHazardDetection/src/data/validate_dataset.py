"""Basic dataset validation helpers."""

from collections import Counter
from pathlib import Path

VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def summarize_image_directory(directory: Path) -> dict[str, object]:
    """Return image counts and extension distribution for a directory tree."""
    if not directory.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {directory}")

    images = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES
    ]
    extension_counts = Counter(path.suffix.lower() for path in images)
    return {
        "directory": str(directory),
        "image_count": len(images),
        "extensions": dict(sorted(extension_counts.items())),
    }
