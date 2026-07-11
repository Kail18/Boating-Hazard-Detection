"""Preprocess all images in a directory while preserving relative paths."""

from argparse import ArgumentParser
from pathlib import Path

from src.processing.preprocess import preprocess_file

VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = [
        path
        for path in args.input.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    ]

    for input_path in image_paths:
        relative_path = input_path.relative_to(args.input)
        output_path = (args.output / relative_path).with_suffix(".png")
        preprocess_file(input_path, output_path)

    print(f"Processed {len(image_paths)} images.")


if __name__ == "__main__":
    main()
