# Repository Setup

The root project is KailMcGuire-CS898BA-Project. The Milestone 1 directory was the project pitch provided at the beginning of the semester.

## Model Repository Setup

```text
KailMcGuire-CS898BA-Project/
├── configs/
│   ├── project.yaml
│   └── yolo_baseline.yaml
│
├── data/
│   └── processed/
│       └── lars_yolo/
│           ├── images/
│           │   ├── train/
│           │   └── val/
│           ├── labels/
│           │   ├── train/
│           │   └── val/
│           ├── dataset.yaml
│           └── import_summary.json
│
├── src/
│   ├── data/
│   │   ├── validate_dataset.py
│   │   └── validate_yolo_dataset.py
│   └── training/
│       └── yolo_trainer.py
│
├── scripts/
│   ├── verify_yolo_dataset.py
│   └── train_yolo.py
│
├── models/
│   └── checkpoints/
│
└── outputs/
    ├── figures/
    ├── metrics/
    └── logs/
```

### Description of each directory

- configs/ stores settings.
- src/ stores reusable program logic.
- scripts/ stores commands you run.
- data/ stores the dataset.
- models/ stores trained weights.
- outputs/ stores results.

#### This is the final model-ready dataset

```text
lars_yolo/
  ├── images/
  │   ├── train/
  │   └── val/
  ├── labels/
  │   ├── train/
  │   └── val/
  ├── dataset.yaml
  └── import_summary.json
```

### Roadblocks and Pivots

- Roadblock - the Ground Truth from the LaRS dataset did not account for unknown object detection. I believe it would be benificial to include this to warn the captain that there is a possible hazard. I will leave the 5 classes from the LaRS dataset for right now since the unknown object is just shoreline clutter. This could get confusing to the captain if its always alerting and can cause the captain to stop paying attention to the detection software.

- Roadblock - the YOLO.pdf research paper mentioned that YOLO struggled identifying smaller objects. It did mention a possible increase in performance when paired with fast R-CNN models. I will implement the models seperate and then combine the two models. I will however be using Faster R-CNN since this is the updated model.

- Roadblock - the smoke test had some images that had multiple boxes for the same object, boxes with no objects, and objects that dont have boxes.
