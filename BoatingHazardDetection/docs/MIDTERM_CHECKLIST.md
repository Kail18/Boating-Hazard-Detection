# Midterm Progress Checklist

The midterm presentation must demonstrate progress, not just describe a future plan.

## 1. Condensed Literature Review

- [ ] Select the core 3-6 papers that directly influence the implementation.
- [ ] Summarize the problem, dataset, architecture, processing, metrics, and findings for each.
- [ ] Explain which ideas are being adopted and which are being rejected.
- [ ] Connect the papers to the selected YOLO baseline and maritime image-processing strategy.

## 2. Dataset and Data Pipeline

- [ ] Finalize the dataset or dataset combination.
- [ ] Document class names and label mappings.
- [ ] Record the number of images and annotations per class.
- [ ] Create train, validation, and test splits without leakage.
- [ ] Save representative raw and processed examples.
- [ ] Document quality issues such as imbalance, small objects, glare, haze, occlusion, and annotation inconsistency.

## 3. Image Processing and Analysis

- [ ] Establish a no-processing control pipeline.
- [ ] Run CLAHE/local contrast enhancement.
- [ ] Run color or intensity normalization.
- [ ] Evaluate at least one visibility-oriented method such as dehazing.
- [ ] Compare histograms or image statistics before and after processing.
- [ ] Explain why each operation is appropriate for maritime images.

## 4. Baseline Model

- [ ] Convert annotations into the required detection format.
- [ ] Train one reproducible YOLO baseline.
- [ ] Save configuration, seed, epoch count, image size, and batch size.
- [ ] Report precision, recall, mAP@0.5, and mAP@0.5:0.95.
- [ ] Save confusion matrix and representative predictions.
- [ ] Identify failure cases by class, object size, lighting, and background.

## 5. Roadblocks and Pivots

- [ ] Record dataset-access or licensing issues.
- [ ] Record label-quality and class-mapping issues.
- [ ] Record hardware, training-time, or memory constraints.
- [ ] Explain any reduction or change in classes.
- [ ] Explain any processing method that was removed after poor results.

## 6. Presentation Package

- [ ] Midterm slide deck.
- [ ] Recorded presentation with audio and video.
- [ ] Repository link.
- [ ] YouTube link set to unlisted or public.
- [ ] Updated `AI_Log.md`.
- [ ] Incremental git history with meaningful commit messages.
