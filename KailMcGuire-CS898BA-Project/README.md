# CS 898BA Maritime Hazard Detection Project

This repository contains a computer vision project for detecting maritime hazards relevant to recreational boating. The current baseline direction uses a YOLO-style object detector combined with deliberate image analysis and preprocessing for difficult water-scene conditions.

## Project Scope

The system is intended to detect visible objects or regions such as:

- Boats
- Buoys
- People in the water
- Floating obstacles
- Docks or piers
- Shoreline or rocks

The project does **not** attempt to implement autonomous navigation, route planning, or vessel control.

## Repository Structure

```text
configs/                 Experiment configuration files
data/
  raw/                   Original dataset files, excluded from git
  interim/               Converted or partially cleaned data
  processed/             Preprocessed model-ready images
  annotations/           Detection labels and mapping files
docs/                    Course requirements, project status, and milestone checklists
models/
  checkpoints/           Training checkpoints, excluded from git
  exported/              Exported inference models, excluded from git
notebooks/               Exploratory analysis notebooks
outputs/
  figures/               Plots and visual comparisons
  predictions/           Saved model predictions
  metrics/               Evaluation reports
  logs/                  Training and processing logs
presentations/           Pitch, midterm, and final presentation materials
research/
  papers/                 Research papers
  notes/                  Paper notes
  lit_review/             Literature synthesis and comparison matrix
scripts/                  Command-line scripts
src/
  data/                   Dataset loading and validation
  processing/             Image-processing pipeline
  training/               Model training code
  evaluation/             Detection metrics and error analysis
  visualization/          Prediction and comparison figures
tests/                    Automated tests
```

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Verify the Project

```bash
python main.py
```

Expected output:

```text
Hello World! CS 898BA Maritime Hazard Detection Project
```

## Import the LaRS Dataset

The selected baseline dataset is LaRS v1.0.0. Download its single-frame images and train/validation annotations from the official dataset page, extract them under `data/raw/lars/`, and run:

```bash
python scripts/import_lars.py
```

This converts the COCO-panoptic segment bounding boxes into Ultralytics YOLO labels and creates `data/processed/lars_yolo/dataset.yaml`. See `docs/DATASET.md` for the selected classes, expected extraction layout, and import options.

## Preprocess Images for Comparison Experiments

The separate image-processing experiment can be run with:

```bash
python scripts/preprocess_dataset.py --input data/raw --output data/processed
```

The initial pipeline applies CLAHE to the LAB luminance channel, resizes images without changing their aspect ratio, and pads them to a consistent model input size.

## Midterm Discussion

During conversion, all 2,803 images were successfully located. One malformed or unusable bounding-box annotation was excluded, while 10,415 valid object annotations were retained.

### Dataset Summary

This is to evaluate the dataset.

From the project root(KailMcGuire-CS898BA-Project) run:

```bash
python - <<'PY'
import json
from pathlib import Path

summary_path = Path("data/processed/lars_yolo/import_summary.json")

with summary_path.open("r", encoding="utf-8") as file:
    summary = json.load(file)

for split_data in summary["splits"]:
    split_name = (
        split_data.get("split")
        or split_data.get("name")
        or split_data.get("split_name")
        or "unknown"
    )

    class_counts = split_data.get("class_counts", {})
    total = sum(class_counts.values())

    print(f"\n{split_name.upper()}")
    print(f"Total objects: {total}")

    for class_name, count in class_counts.items():
        percentage = (count / total * 100) if total else 0
        print(
            f"{class_name:20} "
            f"{count:6} "
            f"({percentage:6.2f}%)"
        )
PY
```

## AI Accountability

All meaningful AI usage must be documented in `AI_Log.md` with the full prompt, timestamp, tool, response synopsis, and resulting design or code change.
