# CS 898BA Midterm Progress Report

## Boating Hazard Detection from Boat-Level Imagery

**Student:** Kail McGuire  
**Course:** CS 898BA — Image Analysis and Computer Vision  
**Project Repository:** [Boating-Hazard-Detection](https://github.com/Kail18/Boating-Hazard-Detection)  
**Current Stage:** Data pipeline complete; YOLO baseline infrastructure and proof-of-concept smoke test complete  
**Date:** July 12, 2026

---

## 1. Project Objective

The goal of this project is to develop a computer-vision system that detects potential boating hazards from boat-level imagery. The current scope is limited to visual hazard detection and does not include autonomous navigation, route planning, or vessel control.

The system currently targets five object classes:

| Class ID | Project Class       |
| -------: | ------------------- |
|        0 | `vessel`            |
|        1 | `buoy`              |
|        2 | `swimmer`           |
|        3 | `paddle_board`      |
|        4 | `floating_obstacle` |

The project is designed around three model experiments:

1. A YOLO one-stage object detector as the initial real-time baseline.
2. A Faster R-CNN two-stage detector as an accuracy and localization comparison.
3. A possible prediction-level ensemble if the two models make complementary errors.

---

## 2. Midterm Requirement Alignment

| Midterm Requirement          | Current Progress                                                                                                                                                  |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Condensed literature review  | Core LaRS, YOLO, Faster R-CNN, and CLAHE sources identified and connected to the project design                                                                   |
| Data pipeline and processing | Complete custom LaRS-to-YOLO conversion, validation, analysis, and visualization pipeline                                                                         |
| Baseline implementation      | YOLO11s training system complete; one-epoch smoke test successfully executed                                                                                      |
| Preliminary results          | Initial validation metrics, prediction examples, checkpoints, and plots generated                                                                                 |
| Roadblocks and pivots        | Dataset format conversion, invalid annotations, class imbalance, tiny objects, ambiguous labels, augmentation effects, and duplicate early predictions documented |
| Full baseline result         | Not yet complete; the full 30-epoch YOLO run is the next major experiment                                                                                         |

---

## 3. Condensed Literature Review

### 3.1 LaRS Maritime Dataset

Žust, Perš, and Kristan introduced LaRS as a diverse maritime obstacle-detection dataset containing imagery from lakes, rivers, and seas. The dataset was selected because it includes varied environmental conditions, scene types, obstacle categories, and recording locations.

LaRS was originally designed for panoptic maritime perception rather than this exact five-class YOLO object-detection task. Therefore, a custom data-conversion pipeline was required to transform the source annotations into normalized YOLO bounding boxes and map the original categories into the project's compact five-class structure.

The dataset is appropriate for this project because it includes realistic challenges such as:

- Small and distant obstacles
- Water glare and reflections
- Different weather and lighting conditions
- Shoreline clutter
- Partial object occlusion
- Significant variation in object scale

### 3.2 YOLO

Redmon et al. presented YOLO as a unified one-stage detector that predicts bounding boxes and class probabilities directly from the complete image in one network evaluation. This design makes YOLO attractive for boating applications where eventual real-time or near-real-time detection is important.

YOLO was selected as the first baseline because it offers:

- Fast inference
- A direct end-to-end detection pipeline
- Straightforward transfer learning
- Established precision, recall, and mAP evaluation
- Practical deployment options

The original YOLO paper also reported weaknesses in precise localization and small-object detection. Those limitations are directly relevant because several LaRS hazards occupy only a small number of pixels.

### 3.3 Faster R-CNN

Ren et al. introduced Faster R-CNN, which uses a Region Proposal Network to identify candidate object regions before classifying and refining them. Unlike YOLO's one-stage design, Faster R-CNN is a two-stage detector.

Faster R-CNN is planned as the second model because it may provide:

- More precise object localization
- Better handling of small or partially visible objects
- A useful speed-versus-accuracy comparison with YOLO

The original YOLO research also reported improved performance when YOLO predictions were combined with Fast R-CNN predictions because the models made different types of errors. For this project, Faster R-CNN will be used as the modern two-stage comparison before deciding whether a combined prediction system is justified.

### 3.4 CLAHE and Planned Image Processing

Contrast Limited Adaptive Histogram Equalization, or CLAHE, improves local contrast while limiting excessive contrast amplification. It is relevant to maritime imagery because hazards may have weak contrast against water, haze, glare, or low-light backgrounds.

CLAHE has not yet been applied to the training dataset. It is planned as a controlled preprocessing experiment after the unprocessed YOLO baseline is established. Other possible experiments include:

- Color or illumination normalization
- Dehazing
- Glare reduction
- Increased training resolution
- Small-object-aware image tiling or cropping

Each preprocessing experiment will be compared against the same baseline rather than being applied without measurement.

---

## 4. Data Pipeline and Processing

The completed data pipeline goes beyond basic augmentation. It performs dataset extraction, category mapping, annotation conversion, numerical validation, class-distribution analysis, and visual ground-truth inspection.

### 4.1 Pipeline Flow

```text
Original LaRS archives
        ↓
Archive extraction
        ↓
Source image and annotation discovery
        ↓
Original category mapping
        ↓
Compact five-class conversion
        ↓
Bounding-box validation
        ↓
YOLO coordinate normalization
        ↓
Train/validation image-label export
        ↓
dataset.yaml generation
        ↓
import_summary.json generation
        ↓
Dataset-wide validation
        ↓
Ground-truth visualization
        ↓
YOLO model training
```

### 4.2 Processed Dataset Structure

```text
data/
└── processed/
    └── lars_yolo/
        ├── images/
        │   ├── train/
        │   └── val/
        ├── labels/
        │   ├── train/
        │   └── val/
        ├── dataset.yaml
        └── import_summary.json
```

### 4.3 Conversion Operations

The conversion pipeline:

- Locates the original LaRS images and annotations
- Maps source categories into the five project classes
- Converts source bounding boxes into YOLO format
- Normalizes box centers, widths, and heights to the range `[0, 1]`
- Creates matching image and label folders
- Generates the YOLO dataset configuration
- Records conversion totals and class distributions
- Skips invalid bounding boxes rather than allowing corrupted labels into training

One invalid bounding-box annotation was identified and skipped during import.

### 4.4 Dataset Validator

A reusable YOLO dataset validator was added under:

```text
src/data/validate_yolo_dataset.py
```

It checks:

- Dataset YAML structure
- Dataset paths
- Class IDs and class order
- Image-label pairing
- Missing label files
- Orphan label files
- Empty background labels
- Numeric annotation values
- Normalized coordinate ranges
- Positive box width and height
- Boxes extending beyond image boundaries
- Agreement with `import_summary.json`

The final validation report contained:

```json
{
  "errors": [],
  "passed": true
}
```

### 4.5 Background Images

The dataset contains empty label files representing images with none of the five target objects:

| Split      | Background Images | Percentage of Split |
| ---------- | ----------------: | ------------------: |
| Training   |               571 |               21.9% |
| Validation |                38 |               19.2% |
| Total      |               609 |               21.7% |

These images were retained because they provide negative examples and can help the model learn not to mistake open water, glare, shoreline, or other clutter for a target hazard.

---

## 5. Dataset Summary

### 5.1 Overall Dataset Size

| Split      |    Images | Labeled Objects |
| ---------- | --------: | --------------: |
| Training   |     2,605 |           9,309 |
| Validation |       198 |           1,106 |
| **Total**  | **2,803** |      **10,415** |

### 5.2 Training Class Distribution

| Class             | Objects | Percentage |
| ----------------- | ------: | ---------: |
| Vessel            |   6,356 |     68.28% |
| Buoy              |   1,566 |     16.82% |
| Swimmer           |     349 |      3.75% |
| Paddle board      |     154 |      1.65% |
| Floating obstacle |     884 |      9.50% |

### 5.3 Validation Class Distribution

| Class             | Objects | Percentage |
| ----------------- | ------: | ---------: |
| Vessel            |     820 |     74.14% |
| Buoy              |     125 |     11.30% |
| Swimmer           |      26 |      2.35% |
| Paddle board      |      30 |      2.71% |
| Floating obstacle |     105 |      9.49% |

### 5.4 Class-Imbalance Finding

The dataset is strongly dominated by vessels. The training split contains approximately 41 vessel annotations for every paddle-board annotation.

Because overall mAP can hide weak performance on rare classes, evaluation will include:

- Per-class precision
- Per-class recall
- Per-class AP
- Overall mAP@0.50
- Overall mAP@0.50:0.95
- Confusion matrices
- Qualitative error analysis
- Small-object performance analysis

---

## 6. Ground-Truth Annotation Verification

A modular visualization pipeline was built to:

- Load an image and its matching label file
- Parse normalized YOLO annotations
- Convert normalized boxes back into pixel coordinates
- Draw class labels and bounding boxes
- Save annotated images for manual review

An initial rendered example showed:

- Two labeled vessels
- One labeled floating obstacle
- A floating-obstacle box measuring approximately `8 × 5` pixels
- A possible additional vessel without a separate annotation

This inspection highlighted two important dataset limitations:

1. Some hazards are extremely small.
2. The source dataset may contain occasional missing or ambiguous labels.

The project currently preserves the original ground truth rather than adding uncertain labels manually. Questionable examples will be documented for error analysis.

---

## 7. Baseline Model Implementation

### 7.1 Current YOLO Architecture

The first model is:

```text
YOLO11s
├── COCO-pretrained weights
├── 640 × 640 model input
├── Five-class detection head
├── Apple MPS training
└── Ultralytics training and validation pipeline
```

### 7.2 Transfer Learning

The model begins with `yolo11s.pt`, which was pretrained on the COCO dataset. Most pretrained parameters were transferred into the custom detector:

```text
Transferred pretrained items: 493 / 499
Original COCO classes:        80
Project classes:               5
```

The detector's final class output was changed from 80 COCO categories to the five maritime project categories.

### 7.3 Model Configuration

| Setting                      | Value                      |
| ---------------------------- | -------------------------- |
| Model                        | YOLO11s                    |
| Pretrained weights           | `yolo11s.pt`               |
| Image size                   | 640                        |
| Seed                         | 42                         |
| Device                       | Apple MPS                  |
| Full baseline epochs         | 30                         |
| Smoke-test epochs            | 1                          |
| Smoke-test batch size        | 4                          |
| Smoke-test training fraction | 5%                         |
| Validation fraction          | 100%                       |
| Model parameters             | Approximately 9.43 million |
| Computation                  | Approximately 21.6 GFLOPs  |

### 7.4 Training Software Structure

The implementation separates responsibilities across:

```text
configs/project.yaml
configs/yolo_baseline.yaml
src/data/validate_yolo_dataset.py
src/training/yolo_trainer.py
scripts/verify_yolo_dataset.py
scripts/train_yolo.py
```

The training system:

- Loads project and model configuration
- Automatically validates the dataset
- Selects CUDA, MPS, or CPU
- Loads pretrained YOLO weights
- Runs smoke or baseline training
- Saves checkpoints
- Records environment metadata
- Copies metrics and figures into organized output directories

---

## 8. Preliminary Baseline Proof of Concept

A one-epoch smoke test was completed before committing to the full baseline.

### 8.1 Smoke-Test Data

| Setting                    | Value |
| -------------------------- | ----: |
| Training images used       |   130 |
| Training background images |     7 |
| Validation images used     |   198 |
| Validation objects         | 1,106 |
| Corrupt images             |     0 |
| Epochs                     |     1 |
| Batch size                 |     4 |

The smoke test successfully completed:

- Dataset loading
- Forward propagation
- Loss calculation
- Backpropagation
- Weight updates
- Full validation
- Checkpoint creation
- Plot generation
- Prediction generation

### 8.2 Preliminary Metrics

| Metric         |          Smoke-Test Result |
| -------------- | -------------------------: |
| Precision      |                      0.675 |
| Recall         |                     0.0461 |
| mAP@0.50       |                     0.0378 |
| mAP@0.50:0.95  |                     0.0195 |
| Inference time | Approximately 9.7 ms/image |

### 8.3 Preliminary Per-Class Results

| Class             | Recall |             AP@0.50 |
| ----------------- | -----: | ------------------: |
| Vessel            |  0.230 |               0.189 |
| Buoy              |  0.000 |               0.000 |
| Swimmer           |  0.000 |               0.000 |
| Paddle board      |  0.000 |               0.000 |
| Floating obstacle |  0.000 | Approximately 0.000 |

These values are not treated as final model performance. The model was trained for only one epoch on 5% of the training split. The purpose of the smoke test was to validate the complete training pipeline.

### 8.4 Generated Outputs

The smoke test generated:

- `best.pt`
- `last.pt`
- `results.csv`
- `results.png`
- `labels.jpg`
- Training batch visualizations
- Validation ground-truth visualizations
- Validation prediction visualizations
- Confusion matrices
- Run configuration
- Environment metadata

---

## 9. Qualitative Preliminary Findings

### 9.1 Small Objects

The model began detecting some smaller objects during the diagnostic smoke testing, but tiny hazards remain a central challenge.

Potential future responses include:

- Higher image resolution
- Reduced mosaic scaling
- Cropping or tiling
- Class-aware sampling
- Longer training
- Comparison with Faster R-CNN

### 9.2 Duplicate Predicted Boxes

Multiple predicted boxes appeared around the same object in:

```text
val_batch0_pred.jpg
val_batch1_pred.jpg
```

These duplicate boxes appeared only in prediction images, not in the ground-truth label visualizations. Therefore, they were identified as early model behavior rather than duplicate annotations.

Likely causes include:

- Only one training epoch
- Very small training subset
- Unstable classification confidence
- Unrefined box localization
- Multiple candidate boxes surviving non-maximum suppression

The dataset was not modified in response to these predictions.

### 9.3 Augmentation Diagnostic

The original smoke run included default geometric and color augmentations such as mosaic, scaling, translation, horizontal flipping, and HSV changes.

Some transformed training previews made tiny objects difficult to see. A diagnostic run with these augmentations disabled made label inspection clearer and qualitatively improved the visibility of smaller targets.

This does not yet prove that augmentation should be removed from the final model. It establishes augmentation strength as an experimental variable that must be evaluated carefully for small maritime hazards.

---

## 10. Roadblocks and Pivots

### 10.1 LaRS Was Not Already in the Required YOLO Format

**Roadblock:** The source dataset was not directly ready for the selected detector.

**Response:** A custom conversion pipeline was created to map classes, validate boxes, normalize coordinates, generate labels, and produce the required YAML configuration.

### 10.2 Invalid Annotation

**Roadblock:** One source bounding box was invalid.

**Response:** The pipeline identified and skipped it rather than passing corrupted data to the model.

### 10.3 Class Imbalance

**Roadblock:** Vessels dominate the dataset, while swimmers and paddle boards are rare.

**Response:** The evaluation design was expanded to include per-class metrics and qualitative analysis rather than relying only on overall mAP.

### 10.4 Extremely Small Hazards

**Roadblock:** Some hazards occupy only a few pixels.

**Response:** Small-object behavior is now a central evaluation criterion. Planned experiments include higher resolution, preprocessing, augmentation adjustment, and comparison with Faster R-CNN.

### 10.5 Missing or Ambiguous Ground Truth

**Roadblock:** Manual review found at least one possible object without a separate label.

**Response:** Uncertain objects were not automatically added to ground truth. They will be documented as a dataset limitation to avoid introducing subjective annotations.

### 10.6 Proposed Unknown-Hazard Class

**Roadblock:** A general unknown-hazard class might warn the operator about unlabeled objects, but shoreline clutter could cause excessive alerts.

**Pivot:** The current implementation retains the five compact classes. Unknown-hazard or anomaly detection remains a possible future extension after the supervised detector is stable.

### 10.7 Augmentation Can Obscure Tiny Objects

**Roadblock:** Mosaic and scaling may reduce already-small objects to only a few visible pixels.

**Response:** Augmentations were temporarily disabled during a diagnostic smoke run. The final augmentation policy will be tested rather than assumed.

### 10.8 Duplicate Early Predictions

**Roadblock:** The smoke model generated multiple boxes on some validation objects.

**Response:** Inspection confirmed that these occurred only in prediction files. The project will first evaluate the fully trained baseline before tuning confidence and non-maximum-suppression settings.

### 10.9 Exact MPS Reproducibility

**Roadblock:** PyTorch reported that one MPS operation does not currently have a fully deterministic implementation.

**Response:** Seed 42 and deterministic settings are retained, but the final report will document that Apple MPS may not provide bit-for-bit identical results across runs.

---

## 11. Current Project Status

| Project Component                    | Status                |
| ------------------------------------ | --------------------- |
| Project scope and research direction | Complete              |
| LaRS dataset acquisition             | Complete              |
| Source archive extraction            | Complete              |
| Five-class mapping                   | Complete              |
| LaRS-to-YOLO conversion              | Complete              |
| Dataset YAML generation              | Complete              |
| Import summary generation            | Complete              |
| Dataset structural validation        | Complete              |
| Class-distribution analysis          | Complete              |
| Initial ground-truth visualization   | Complete              |
| Broader manual annotation audit      | In progress           |
| YOLO model configuration             | Complete              |
| YOLO training script                 | Complete              |
| YOLO smoke test                      | Complete              |
| Preliminary prediction review        | Complete              |
| Full 30-epoch YOLO baseline          | Not started           |
| Full YOLO baseline evaluation        | Not started           |
| Faster R-CNN implementation          | Not started           |
| CLAHE and preprocessing experiments  | Not started           |
| Hyperparameter optimization          | Not started           |
| YOLO/Faster R-CNN comparison         | Not started           |
| Prediction-level ensemble experiment | Optional future stage |
| Final demonstration                  | Not started           |

---

## 12. Next Steps

### Immediate Next Step

Run the complete YOLO11s baseline using the full training split:

```bash
python scripts/train_yolo.py --mode baseline
```

Before the run, the final batch size and augmentation settings should be frozen and documented.

### Baseline Evaluation

After training:

1. Record overall precision and recall.
2. Record mAP@0.50 and mAP@0.50:0.95.
3. Record per-class AP and recall.
4. Review the confusion matrix.
5. Inspect false positives and false negatives.
6. Analyze duplicate detections.
7. Analyze failures on tiny objects.
8. Measure inference speed.
9. Compare `best.pt` with `last.pt`.

### Later Experiments

1. Implement Faster R-CNN with a ResNet-50 FPN backbone.
2. Compare one-stage and two-stage performance.
3. Test CLAHE and other preprocessing methods.
4. Test higher input resolution.
5. Test augmentation changes for tiny objects.
6. Determine whether the two detectors make complementary errors.
7. Test prediction fusion only if justified by the results.

---

## 13. Recommended Midterm Presentation Evidence

The presentation should include screenshots or figures showing:

1. Original LaRS image
2. Converted ground-truth bounding boxes
3. Dataset directory structure
4. `dataset.yaml` class mapping
5. Train and validation class-distribution table
6. Example of the approximately `8 × 5` pixel obstacle
7. Dataset validator output showing `passed: true`
8. YOLO11s model summary
9. Smoke-test loss and metric plots
10. `val_batch*_labels.jpg`
11. `val_batch*_pred.jpg`
12. Example of duplicate early predictions
13. Current status and next-step table

Suggested paths include:

```text
outputs/figures/label_verification/
outputs/logs/yolo/yolo11s_compact5_smoke/
outputs/metrics/yolo/
models/checkpoints/yolo/
```

---

## 14. Midterm Conclusion

The project has progressed from an approved concept into a functioning end-to-end object-detection pipeline.

The completed work includes:

- A custom LaRS-to-YOLO conversion system
- A validated five-class maritime dataset
- Dataset quality and class-imbalance analysis
- Ground-truth visualization
- A modular YOLO training architecture
- Transfer learning from pretrained YOLO11s weights
- Successful Apple MPS training
- A completed smoke test with checkpoints, metrics, and predictions
- Initial diagnosis of tiny-object, augmentation, and duplicate-prediction behavior

The project has achieved the midterm proof-of-concept objective. The next milestone is to execute and evaluate the complete 30-epoch YOLO baseline, followed by the Faster R-CNN comparison and controlled image-processing experiments.

---

## References

1. L. Žust, J. Perš, and M. Kristan, “LaRS: A Diverse Panoptic Maritime Obstacle Detection Dataset and Benchmark,” _Proceedings of the IEEE/CVF International Conference on Computer Vision_, 2023.

2. J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You Only Look Once: Unified, Real-Time Object Detection,” _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_, 2016.

3. S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks,” _Advances in Neural Information Processing Systems_, 2015.

4. K. Zuiderveld, “Contrast Limited Adaptive Histogram Equalization,” in _Graphics Gems IV_, Academic Press, 1994, pp. 474–485.
