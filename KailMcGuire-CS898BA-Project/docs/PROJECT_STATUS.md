# Project Status

## Current Direction

**Working title:** Computer Vision-Based Maritime Hazard Detection for Recreational Boating

**Primary goal:** Detect visible hazards that could matter to a recreational boater, including boats, buoys, people in the water, floating obstacles, docks or piers, and shoreline or rocks.

**Scope boundary:** This project is an object-detection and image-analysis system. It is not a complete autonomous-navigation, route-planning, collision-avoidance, or vessel-control system.

## Current Technical Direction

- Primary architecture: a YOLO-style object detector.
- Alternative design 1: a two-stage detector such as Faster R-CNN.
- Alternative design 2: an image-level classifier or segmentation-first pipeline.
- Current justification: YOLO provides a practical balance between detection accuracy, training complexity, and real-time inference potential for a semester project.

## Required Domain Engineering

The image-processing component must go beyond generic augmentation. Initial experiments will compare:

1. CLAHE/local contrast enhancement for glare, shadows, and uneven lighting.
2. Color and intensity normalization for changing water and sky conditions.
3. Optional dehazing or visibility enhancement for haze and spray.
4. Edge and texture analysis to document how small hazards differ from water background.
5. Aspect-ratio-preserving resizing and padding before model training.

## Repository Status on July 11, 2026

- Course requirements preserved in `docs/course_requirements.md`.
- Python package and project directories created.
- AI accountability log created.
- Baseline preprocessing utilities created.
- Dataset has not yet been added.
- Training has not yet been executed.
- Baseline metrics and prediction examples are still pending.
- Condensed literature review is still pending consolidation into the repository.
