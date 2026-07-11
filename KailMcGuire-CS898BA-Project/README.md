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

## Midterm Objective

By the midterm milestone, the repository should contain:

1. A condensed literature review tied directly to the implementation.
2. A documented dataset and image-processing pipeline.
3. A reproducible baseline detector with preliminary metrics.
4. Prediction examples and failure-case analysis.
5. A discussion of roadblocks and any scope pivots.

See `docs/MIDTERM_CHECKLIST.md` for the working checklist and `docs/PROJECT_STATUS.md` for the current state.

## AI Accountability

All meaningful AI usage must be documented in `AI_Log.md` with the full prompt, timestamp, tool, response synopsis, and resulting design or code change.
