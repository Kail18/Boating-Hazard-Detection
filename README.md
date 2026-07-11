# Repository Setup

The root project is KailMcGuire-CS898BA-Project. The Milestone 1 directory was the project pitch provided at the beginning of the semester.

## Midterm Progress Report

The project focuses on detecting maritime hazards from boat-level imagery using a YOLO-style object detector. The current scope is object detection only and does not include autonomous navigation or vessel control.

### Completed

- Selected the LaRS maritime dataset
- Downloaded and extracted the image and annotation archives
- Created a custom LaRS-to-YOLO conversion pipeline
- Mapped the original annotations into five project classes:

  - `vessel`
  - `buoy`
  - `swimmer`
  - `paddle_board`
  - `floating_obstacle`

- Converted 2,803 images and 10,415 object annotations into YOLO format
- Generated the required `dataset.yaml` file
- Verified that no referenced images were missing
- Identified and skipped one invalid bounding box
- Analyzed the class distribution and identified substantial class imbalance

### Dataset Summary

| Split      | Images | Objects |
| ---------- | -----: | ------: |
| Training   |  2,605 |   9,309 |
| Validation |    198 |   1,106 |

The dataset is dominated by vessels, while swimmers and paddle boards are underrepresented. Because of this imbalance, the project will report both overall and per-class model performance.

### Next Steps

- Visualize sample bounding boxes to confirm annotation accuracy
- Train an unprocessed YOLO baseline
- Evaluate precision, recall, mAP, confusion matrices, and per-class performance
- Analyze missed detections and false positives
- Test image-processing methods such as CLAHE, color normalization, dehazing, and higher image resolution
- Compare each experiment against the baseline
- Complete the literature review and midterm presentation
