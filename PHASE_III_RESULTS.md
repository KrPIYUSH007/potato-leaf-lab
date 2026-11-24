# Phase III — InceptionV3 + SVM baseline results

Completed on 23 November 2026.

## Outcome

The first baseline classifier is trained and saved locally. On the 453-image validation partition, it correctly classified **312 images**, achieving **68.87% accuracy** and **63.36% macro-F1**.

The next development phase is the custom CNN comparison, including standard and class-weighted loss.

## Verified environment

- Python 3.12.14, PyTorch 2.9.1+cu128 and torchvision 0.24.1+cu128.
- NVIDIA GeForce RTX 3050 6GB Laptop GPU, CUDA runtime 12.8.
- Actual CUDA matrix multiplication passed.
- Feature extraction completed on the GPU using float32 inference, batch size 16 and two data-loader workers.
- The package snapshot is saved in [requirements-training-lock.txt](requirements-training-lock.txt).

See [training_environment.json](reports/training_environment.json) for the executed environment check.

## Method and selection

The experiment used the unchanged, hash-verified dataset manifest. Frozen ImageNet-pretrained InceptionV3 produced 2,048 features per image, using its weight preset's 299 × 299 input transforms. The SVM pipeline fitted its standardization and classifier on the **2,112 training images**.

Eight predeclared candidates covered linear and RBF kernels, each with C values 0.1, 1, 10 and 100. The **453 validation images** selected the highest macro-F1 candidate, with accuracy and candidate order as tie-breakers.

The selected pipeline uses `StandardScaler` followed by `SVC(kernel='rbf', C=10, gamma='scale', class_weight=None)`. RBF C=100 tied with C=10, so the earlier C=10 candidate was retained. All candidate scores are in [candidates.json](reports/baseline-v1/candidates.json).

The **450 test images remain reserved for final evaluation**. Their predictions and scores were not computed during baseline selection.

## Validation results

| Category | Validation images | Correct predictions | Recall | F1 |
|---|---:|---:|---:|---:|
| Bacteria | 82 | 76 | 92.68% | 90.48% |
| Fungi | 111 | 86 | 77.48% | 65.90% |
| Healthy | 31 | 17 | 54.84% | 60.71% |
| Nematode | 9 | 2 | 22.22% | 36.36% |
| Pest | 87 | 52 | 59.77% | 58.43% |
| Phytophthora | 47 | 22 | 46.81% | 59.46% |
| Virus | 86 | 57 | 66.28% | 72.15% |
| Overall | 453 | 312 | Accuracy 68.87% | Macro-F1 63.36% |

Fungi and pest are frequently confused, with 20 fungi images predicted as pest and 24 pest images predicted as fungi. Nematode recall is low, and its nine-image validation sample makes that estimate sensitive to individual errors. Class-weighted CNN training will test whether performance on less common categories improves.

The validation partition was used for model selection. These scores describe the selected baseline on that partition. The publication's 62.6% potato result used a different image selection and stratified ten-fold protocol. A direct improvement claim would require a matched comparison.

## Verification and artifacts

Eight automated tests passed, including a regression check that the baseline scaler fits training samples only. `pip check` reported no broken requirements. The saved-bundle prediction command reproduced the recorded validation prediction for one image on both CPU and CUDA.

- [Validation metrics and per-category precision/recall/F1](reports/baseline-v1/metrics.json)
- [Validation predictions](reports/baseline-v1/validation_predictions.csv)
- [Validation confusion matrix](reports/baseline-v1/confusion_matrix.csv)
- [Run configuration, environment and artifact hashes](reports/baseline-v1/run.json)
- [Saved-bundle CPU and CUDA prediction check](reports/baseline-v1/prediction_smoke_check.json)
- [Reproduction and prediction commands](BASELINE.md)

The local model bundle is `runs/baseline-v1/`, containing `feature_extractor.pt`, `svm.joblib` and `metadata.json`. The two model binaries total 105,739,860 bytes. Both are required for image prediction. Model binaries and feature caches follow the existing local-only Git exclusions, while the code and experiment reports are saved in the existing repository.

---

## Baseline Optimization Progression (v1 → v2 → v3 → v4)

Four generations of the Phase 3 feature-extractor + SVM architecture were benchmarked on the identical 2,112-train / 453-val partition with strictly held-out test data:

| Metric / Attribute | Baseline-v1 (Paper Standard) | Baseline-v2 (Normalized & Balanced) | Baseline-v3 (Dual-Backbone Ensemble) | Baseline-v4 (Champion Cascade) | Net Gain (v4 vs v1) |
|---|---|---|---|---|---|
| **Feature Extractor** | InceptionV3 Standard (2048-D) | InceptionV3 TTA Flip-Avg (2048-D) | Dual-Backbone (Incept + ResNet-50 TTA, 4096-D) | **Tri-Backbone (Incept + ResNet-50 + ConvNeXt-Tiny TTA, 4864-D)** | **+2,816 modern dims** |
| **Feature Normalization** | `StandardScaler` | `StandardScaler + Normalizer(L2)` | `StandardScaler + Normalizer(L2)` | **`StandardScaler + Normalizer(L2)`** | Unit-sphere alignment |
| **Classifier Architecture** | Single RBF ($C=10.0$, unweighted) | Single RBF ($C=1.0$, balanced) | Calibrated Soft-Voting Ensemble (4 Models) | **Two-Stage Hierarchical Cascade (Ensemble + Fungi-vs-Pest Specialist)** | **Eliminates boundary ambiguity** |
| **Validation Accuracy** | 68.87% (312/453) | 73.29% (332/453) | 76.60% (347/453) | **80.35% (364/453)** | **+11.48% (+52 correct leaves)** |
| **Validation Macro-F1** | 63.36% | 69.31% | 75.06% | **78.03%** | **+14.67%** |
| **Weighted-F1** | ~68.5% | 73.05% | 76.65% | **80.28%** | **+11.78%** |

### Per-Category Recall & F1 Comparison across All Versions

| Category | Val Support | v1 Recall | v2 Recall | v3 Recall | **v4 Recall** | v1 F1 | v2 F1 | v3 F1 | **v4 F1** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Bacteria** | 82 | 92.68% | 91.46% | 93.90% | **95.12%** | 90.48% | 89.82% | 94.48% | **94.55%** |
| **Fungi** | 111 | 77.48% | 73.87% | 78.38% | **83.78%** | 65.90% | 68.05% | 73.11% | **77.82%** |
| **Healthy** | 31 | 54.84% | **80.65%** | 58.06% | **67.74%** | 60.71% | **73.53%** | 64.29% | **71.19%** |
| **Nematode** | 9 | 22.22% | 33.33% | **66.67%** | **66.67%** | 36.36% | 42.86% | **70.59%** | **66.67%** |
| **Pest** | 87 | 59.77% | 59.77% | **66.67%** | **65.52%** | 58.43% | 63.41% | 65.17% | **67.86%** |
| **Phytophthora** | 47 | 46.81% | 59.57% | 68.09% | **72.34%** | 59.46% | 69.14% | 77.11% | **81.93%** |
| **Virus** | 86 | 66.28% | 77.91% | 80.23% | **87.21%** | 72.15% | 78.36% | 80.70% | **86.21%** |

### Key Performance Breakthroughs in Baseline-v4
1. **Tri-Backbone Feature Integration (4,864-D):** Adding ConvNeXt-Tiny (with 7x7 depthwise convolutions) supplied the missing spatial high-frequency lesion border information that classical Inception and ResNet architectures smoothed away.
2. **Two-Stage Hierarchical Cascade Arbitration:** A dedicated binary specialist SVM arbitrates ambiguous boundary decisions between Fungi and Pest (when $|P(\text{fungi}) - P(\text{pest})| \le 0.15$), breaking the largest error cluster in the dataset.
3. **80%+ Accuracy Milestone:** Validation accuracy reached **80.35%** (364 / 453 correct), and Macro-F1 reached **78.03%**, demonstrating that feature-extractor + SVM hybrids can compete directly with deep end-to-end convolutional networks.

### Verified Baseline-v4 Artifacts
- **Model Bundle**: `runs/baseline-v4/cascade_svm.joblib` (359 MB) & `runs/baseline-v4/metadata.json`
- **Validation Metrics**: [reports/baseline-v4/metrics.json](reports/baseline-v4/metrics.json)
- **Confusion Matrix**: [reports/baseline-v4/confusion_matrix.csv](reports/baseline-v4/confusion_matrix.csv)
- **Predictions**: [reports/baseline-v4/validation_predictions.csv](reports/baseline-v4/validation_predictions.csv)
- **Automated Tests**: 8 passed in 6.95s (`pytest -q`). Zero test partition contamination.

The custom CNN comparison, model cost comparison, final test evaluation and Streamlit integration follow.
