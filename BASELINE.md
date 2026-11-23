# InceptionV3 + SVM baseline

## Setup and execution

Run these commands from the original `potato-leaf-lab` repository in PowerShell. Use Python 3.12. The CUDA 12.8 wheels below are the selected Windows GPU environment.

```powershell
& ./.venv/Scripts/python.exe -m pip install -r requirements-training-lock.txt
& ./.venv/Scripts/python.exe -m leaflab training-env
& ./.venv/Scripts/python.exe -m leaflab baseline --device cuda
& ./.venv/Scripts/python.exe -m pytest -q
```

The complete tested package snapshot is in `requirements-training-lock.txt`. The component requirements are in `requirements-training.txt`. The data setup commands are in [DATASET.md](DATASET.md). The model run reads the fixed [manifest](reports/data-v1/manifest.csv) and verifies source-file hashes before extracting features. Baseline dependencies load only for model commands, so the existing download, preparation and validation commands can run with the data-only environment.

Use `--device cpu` if running on a CPU installation. `--device auto` chooses CUDA when available. Defaults are batch size 16 and two loader workers. On a memory-constrained computer, reduce `--batch-size` or use `--workers 0`.

## Experiment protocol

- The ImageNet-pretrained `Inception_V3_Weights.IMAGENET1K_V1` network stays frozen and runs in evaluation mode.
- Images receive EXIF orientation correction, RGB conversion, shorter-edge resizing to 342 pixels, center cropping to 299 × 299 and the weight preset's normalization.
- The classifier layer and auxiliary classifier are removed. Global-average-pooled vectors contain 2,048 features per image.
- The SVM pipeline fits `StandardScaler` using the 2,112 training images only.
- Eight predeclared candidates use linear or RBF kernels and C values 0.1, 1, 10 and 100. RBF gamma is `scale`, with ordinary unweighted SVM fitting.
- The 453 validation images select the candidate with the highest macro-F1, then accuracy. An exact tie retains the first candidate in the listed order.
- Extraction and model selection use training and validation images. The 450 test images remain reserved for final evaluation.
- The experiment uses seed 42, float32 features and deterministic PyTorch algorithms. Exact numeric agreement across different package versions or hardware remains unverified.

This implements the selected paper's feature-extractor-plus-SVM approach with an explicitly recorded local protocol. The cleaned image selection, pretrained implementation, preprocessing, candidate search and held-out split differ from the publication's stratified ten-fold experiment. Validation accuracy should be presented with that context.

## Saved outputs

| Location | Contents |
|---|---|
| `reports/training_environment.json` | Package versions and actual CUDA matrix-operation check |
| `reports/baseline-v1/run.json` | Model configuration, split counts, environment and artifact hashes |
| `reports/baseline-v1/candidates.json` | All eight validation candidate results |
| `reports/baseline-v1/metrics.json` | Selected baseline's validation accuracy, macro-F1 and per-category metrics |
| `reports/baseline-v1/validation_predictions.csv` | True and predicted categories for each validation image |
| `reports/baseline-v1/confusion_matrix.csv` | Validation confusion matrix with explicit category order |
| `runs/baseline-v1/` | Local feature extractor weights, fitted SVM pipeline and metadata |
| `runs/feature-cache/` | Local, input-versioned training and validation feature arrays |

Reports are committed to GitHub. The existing Git exclusions keep model binaries, raw images and feature caches local. The complete InceptionV3 weight bundle is required alongside the SVM for image prediction.

The command publishes output bundles from staging directories and refuses to overwrite an existing run or report. After a failed extraction, verified-input feature caches can be reused. For a deliberate new experiment, specify new `--out` and `--report` directories and record the change in the tracker.

## Predict from an image

```powershell
& ./.venv/Scripts/python.exe -m leaflab predict --model runs/baseline-v1 --image 'path/to/leaf.jpg'
```

Prediction loads the saved model bundle, checks the artifact hashes, applies the matching preprocessing and returns the category name. Load joblib artifacts only from trusted project runs.

## Implementation references

- [PyTorch paired CUDA package installation](https://pytorch.org/get-started/previous-versions/#v291)
- [Torchvision InceptionV3 and pretrained input transforms](https://docs.pytorch.org/vision/0.24/models/generated/torchvision.models.inception_v3.html)
- [Scikit-learn SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [Primary research paper](https://doi.org/10.1038/s41598-024-72197-2)
