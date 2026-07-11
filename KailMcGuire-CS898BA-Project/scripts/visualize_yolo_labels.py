"""
Coordinate the incremental YOLO annotation visualization workflow.

The detailed implementation is stored in ``src/visualization_yolo``. This
script coordinates the completed visualization stages and prints readable
verification results.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.visualization_yolo import (
    YoloAnnotationLocator,
    YoloAnnotationParser,
    YoloAnnotationRenderer,
    YoloDatasetVerifier,
)

DEFAULT_DATASET_YAML = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lars_yolo"
    / "dataset.yaml"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "label_verification"
)


def main() -> None:
    """
    Run the completed YOLO visualization preparation stages.

    This coordinator intentionally contains very little processing logic.
    Each major responsibility belongs to a dedicated class under
    ``src/visualization_yolo``.

    The current workflow:

    1. Validate the dataset YAML and image directories.
    2. Select one training image with a non-empty label file.
    3. Parse each annotation and calculate its pixel bounding box.
    4. Print the intermediate results for manual inspection.

    No bounding boxes are drawn or saved during this version.
    """

    # Stage 1: Verify the YOLO dataset.
    dataset_verifier = YoloDatasetVerifier(
        dataset_yaml=DEFAULT_DATASET_YAML
    )

    dataset_result = dataset_verifier.verify()

    print("YOLO Dataset Verification")
    print("-------------------------")
    print(f"Configuration: {DEFAULT_DATASET_YAML}")
    print(f"Dataset root: {dataset_result.dataset_root}")
    print(
        f"Training images: "
        f"{dataset_result.train_image_count}"
    )
    print(
        f"Validation images: "
        f"{dataset_result.validation_image_count}"
    )

    print("\nConfigured classes:")

    for class_id, class_name in (
        dataset_result.class_names.items()
    ):
        print(f"  {class_id}: {class_name}")

    print(
        "\nStage 1 dataset verification "
        "completed successfully."
    )

    # Stage 2: Locate one annotated image-label pair.
    annotation_locator = YoloAnnotationLocator(
        dataset_root=dataset_result.dataset_root,
        image_directory=dataset_result.train_directory,
        split_name="train",
    )

    pair_result = (
        annotation_locator.find_annotated_pair()
    )

    print("\nSample annotated image:")
    print(f"  Image: {pair_result.image_path}")
    print(f"  Label: {pair_result.label_path}")
    print(
        f"  Objects in label file: "
        f"{len(pair_result.annotation_lines)}"
    )

    print(
        "\nStage 2 image-label pairing "
        "completed successfully."
    )

    # Stage 3: Parse annotations and calculate pixel boxes.
    annotation_parser = YoloAnnotationParser(
        class_names=dataset_result.class_names
    )

    image_width, image_height = (
        annotation_parser.load_image_dimensions(
            pair_result.image_path
        )
    )

    parsed_annotations = (
        annotation_parser.parse_lines(
            pair_result.annotation_lines
        )
    )

    print("\nImage dimensions:")
    print(f"  Width: {image_width} pixels")
    print(f"  Height: {image_height} pixels")

    print("\nParsed annotations and pixel coordinates:")

    for annotation_number, annotation in enumerate(
        parsed_annotations,
        start=1,
    ):
        left, top, right, bottom = (
            annotation_parser.convert_to_pixel_box(
                annotation=annotation,
                image_width=image_width,
                image_height=image_height,
            )
        )

        pixel_width = right - left
        pixel_height = bottom - top

        print(f"\n  Object {annotation_number}")
        print(
            f"    Class: {annotation.class_id} "
            f"({annotation.class_name})"
        )
        print(
            "    Normalized box: "
            f"center=({annotation.center_x:.6f}, "
            f"{annotation.center_y:.6f}), "
            f"size=({annotation.width:.6f}, "
            f"{annotation.height:.6f})"
        )
        print(
            "    Pixel corners: "
            f"left={left}, top={top}, "
            f"right={right}, bottom={bottom}"
        )
        print(
            f"    Pixel size: "
            f"{pixel_width} × {pixel_height}"
        )

    # This message belongs outside the loop so it prints only once.
    print(
        "\nStage 3 coordinate conversion "
        "completed successfully."
    )

        # Stage 4: Draw and save the sample ground-truth annotations.
    annotation_renderer = YoloAnnotationRenderer(
        output_directory=DEFAULT_OUTPUT_DIRECTORY
    )

    rendered_result = (
        annotation_renderer.render_annotations(
            image_path=pair_result.image_path,
            annotations=parsed_annotations,
            annotation_parser=annotation_parser,
        )
    )

    print("\nRendered annotation visualization:")
    print(
        f"  Source image: "
        f"{rendered_result.source_image_path}"
    )
    print(
        f"  Saved image: "
        f"{rendered_result.output_image_path}"
    )
    print(
        f"  Boxes drawn: "
        f"{rendered_result.object_count}"
    )
    print(
        f"  Image size: "
        f"{rendered_result.image_width} × "
        f"{rendered_result.image_height}"
    )

    print(
        "\nStage 4 bounding-box rendering "
        "completed successfully."
    )


if __name__ == "__main__":
    main()