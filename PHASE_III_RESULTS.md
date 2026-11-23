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

The custom CNN, model cost comparison, final test evaluation and Streamlit integration remain pending.
