"""Extract ResNet-50 features, fuse with InceptionV3, and train max-accuracy Phase 3 SVM ensemble."""
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set environment and initialize Windows C runtime
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import leaflab.prepare
import torch
from PIL import Image, ImageOps
import joblib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50

from leaflab.download import CLASSES
from leaflab.prepare import validate_manifest

WEIGHTS = ResNet50_Weights.IMAGENET1K_V2


class StepTimer:
    def __init__(self, name):
        self.name = name
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        now_str = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'='*80}", flush=True)
        print(f"[{now_str}] >>> STEP START: {self.name}", flush=True)
        print(f"{'='*80}", flush=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start
        now_str = datetime.now().strftime("%H:%M:%S")
        print(f"[{now_str}] <<< STEP COMPLETED: {self.name} in {self.elapsed:.2f}s ({self.elapsed/60:.2f} min)", flush=True)
        print(f"{'='*80}\n", flush=True)


class ResNetDataset(Dataset):
    def __init__(self, rows, root, tta=False):
        self.rows = rows
        self.root = Path(root)
        self.transform = WEIGHTS.transforms()
        self.tta = tta

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path = self.root / self.rows[index]['path']
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            if not self.tta:
                return self.transform(image)
            t1 = self.transform(image)
            t2 = self.transform(ImageOps.mirror(image))
            return torch.stack([t1, t2])


def get_resnet_extractor(device):
    model = resnet50(weights=WEIGHTS)
    model.fc = torch.nn.Identity()
    model.requires_grad_(False)
    return model.eval().to(device)


def extract_resnet(model, rows, root, device, batch_size=32, tta=False, desc=""):
    loader = DataLoader(
        ResNetDataset(rows, root, tta=tta),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 0 workers avoids Windows multiprocessing fork overhead
        pin_memory=(device.type == 'cuda')
    )
    features = []
    started = time.perf_counter()
    total_imgs = len(rows)

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            b_start = time.perf_counter()
            if not tta:
                output = model(batch.to(device, non_blocking=(device.type == 'cuda')))
                features.append(output.cpu().numpy())
            else:
                b, n, c, h, w = batch.shape
                flat = batch.view(b * n, c, h, w).to(device, non_blocking=(device.type == 'cuda'))
                out = model(flat).view(b, n, -1).mean(dim=1)
                features.append(out.cpu().numpy())

            processed = min((batch_index + 1) * batch_size, total_imgs)
            elapsed = time.perf_counter() - started
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total_imgs - processed) / rate if rate > 0 else 0

            if batch_index % 15 == 0 or processed == total_imgs:
                now_str = datetime.now().strftime("%H:%M:%S")
                print(f"[{now_str}] [{desc}] {processed}/{total_imgs} ({processed/total_imgs*100:5.1f}%) | "
                      f"Elapsed: {elapsed:5.1f}s | Rate: {rate:4.1f} img/s | ETA: {eta:4.1f}s", flush=True)

    result = np.concatenate(features)
    if result.shape != (len(rows), 2048) or not np.isfinite(result).all():
        raise ValueError(f"Invalid ResNet50 features shape {result.shape}")
    return result


def main():
    overall_start = time.perf_counter()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing Phase 3 Maximum SVM Optimization Pipeline...", flush=True)

    root = Path('data/raw')
    manifest_path = Path('reports/data-v1/manifest.csv')
    cache_dir = Path('runs/feature-cache')
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Manifest Verification
    with StepTimer("Manifest & Split Loading"):
        with manifest_path.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        train_rows = [r for r in rows if r['split'] == 'train']
        val_rows = [r for r in rows if r['split'] == 'val']
        y_train = np.array([r['label'] for r in train_rows])
        y_val = np.array([r['label'] for r in val_rows])
        print(f"Loaded: {len(train_rows)} train images, {len(val_rows)} val images across 7 classes.", flush=True)

    # 2. ResNet-50 Feature Extraction & Cache
    res_std_tr_p = cache_dir / 'resnet50_train.npy'
    res_std_va_p = cache_dir / 'resnet50_val.npy'
    res_tta_tr_p = cache_dir / 'resnet50_tta_train.npy'
    res_tta_va_p = cache_dir / 'resnet50_tta_val.npy'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computing on hardware device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    if not (res_std_tr_p.exists() and res_std_va_p.exists() and res_tta_tr_p.exists() and res_tta_va_p.exists()):
        with StepTimer("ResNet-50 Deep Feature Extraction (GPU)"):
            res_model = get_resnet_extractor(device)
            print("1/4: ResNet50 Standard Train (2,112 images)...", flush=True)
            X_tr_res_std = extract_resnet(res_model, train_rows, root, device, tta=False, desc="ResNet50 Std Train")
            np.save(res_std_tr_p, X_tr_res_std)

            print("2/4: ResNet50 Standard Val (453 images)...", flush=True)
            X_va_res_std = extract_resnet(res_model, val_rows, root, device, tta=False, desc="ResNet50 Std Val")
            np.save(res_std_va_p, X_va_res_std)

            print("3/4: ResNet50 TTA Train (2,112 images x 2 flips)...", flush=True)
            X_tr_res_tta = extract_resnet(res_model, train_rows, root, device, tta=True, desc="ResNet50 TTA Train")
            np.save(res_tta_tr_p, X_tr_res_tta)

            print("4/4: ResNet50 TTA Val (453 images x 2 flips)...", flush=True)
            X_va_res_tta = extract_resnet(res_model, val_rows, root, device, tta=True, desc="ResNet50 TTA Val")
            np.save(res_tta_va_p, X_va_res_tta)
            print("ResNet-50 feature matrices successfully cached to disk.", flush=True)
    else:
        with StepTimer("Loading Cached ResNet-50 Features"):
            X_tr_res_std = np.load(res_std_tr_p)
            X_va_res_std = np.load(res_std_va_p)
            X_tr_res_tta = np.load(res_tta_tr_p)
            X_va_res_tta = np.load(res_tta_va_p)
            print(f"ResNet-50 loaded: Train shape={X_tr_res_std.shape}, Val shape={X_va_res_std.shape}", flush=True)

    # 3. Load InceptionV3 and Create Dual-Backbone Fused Embeddings
    with StepTimer("Dual-Backbone Feature Normalization & Fusion"):
        X_tr_incept_std = np.load(cache_dir / 'standard_train.npy')
        X_va_incept_std = np.load(cache_dir / 'standard_val.npy')
        X_tr_incept_tta = np.load(cache_dir / 'tta_train.npy')
        X_va_incept_tta = np.load(cache_dir / 'tta_val.npy')

        # Project both to unit hypersphere prior to concatenation
        l2_norm = Normalizer(norm='l2')
        fused_tr_std = np.hstack([l2_norm.transform(X_tr_incept_std), l2_norm.transform(X_tr_res_std)])
        fused_va_std = np.hstack([l2_norm.transform(X_va_incept_std), l2_norm.transform(X_va_res_std)])

        fused_tr_tta = np.hstack([l2_norm.transform(X_tr_incept_tta), l2_norm.transform(X_tr_res_tta)])
        fused_va_tta = np.hstack([l2_norm.transform(X_va_incept_tta), l2_norm.transform(X_va_res_tta)])

        print(f"Fused Standard Features shape: Train {fused_tr_std.shape}, Val {fused_va_std.shape}", flush=True)
        print(f"Fused TTA Features shape:      Train {fused_tr_tta.shape}, Val {fused_va_tta.shape}", flush=True)

    # 4. Benchmarking Feature Representations & Architectures
    with StepTimer("Benchmarking Multi-Architecture Feature Spaces"):
        feature_spaces = [
            ("InceptionV3 TTA (2048-D)", X_tr_incept_tta, X_va_incept_tta),
            ("ResNet50 TTA (2048-D)", X_tr_res_tta, X_va_res_tta),
            ("Dual-Backbone Fused TTA (4096-D)", fused_tr_tta, fused_va_tta),
        ]

        leaderboard = []

        for space_name, X_tr, X_va in feature_spaces:
            sp_start = time.perf_counter()
            scaler = make_pipeline(StandardScaler(), Normalizer(norm='l2'))
            X_tr_s = scaler.fit_transform(X_tr)
            X_va_s = scaler.transform(X_va)

            print(f"\n--- Testing Space: {space_name} ---", flush=True)
            for C_val in [0.5, 1.0, 2.0, 3.0, 5.0]:
                for cw in ['balanced', None]:
                    cw_label = 'balanced' if cw == 'balanced' else 'unweighted'
                    m_start = time.perf_counter()
                    clf = SVC(kernel='rbf', C=C_val, gamma='scale', class_weight=cw, cache_size=512)
                    clf.fit(X_tr_s, y_train)
                    preds = clf.predict(X_va_s)
                    acc = float(accuracy_score(y_val, preds))
                    mf1 = float(f1_score(y_val, preds, labels=list(CLASSES), average='macro', zero_division=0))
                    duration = time.perf_counter() - m_start
                    leaderboard.append({
                        'space': space_name,
                        'model': f"RBF C={C_val} {cw_label}",
                        'accuracy': acc,
                        'macro_f1': mf1,
                        'duration_s': duration
                    })
                    print(f"  [{space_name:32s}] RBF C={C_val:3.1f} {cw_label:10s} -> "
                          f"Val Acc: {acc*100:5.2f}% | Macro-F1: {mf1*100:5.2f}% ({duration:.2f}s)", flush=True)

            # Test Supervised LDA on this space
            lda_timer = time.perf_counter()
            lda = LinearDiscriminantAnalysis()
            tr_lda = lda.fit_transform(X_tr_s, y_train)
            va_lda = lda.transform(X_va_s)

            # Combine LDA coordinates (6-D) + L2 scaled features
            lda_combined_tr = Normalizer(norm='l2').fit_transform(np.hstack([tr_lda, X_tr_s]))
            lda_combined_va = Normalizer(norm='l2').transform(np.hstack([va_lda, X_va_s]))

            for C_val in [1.0, 2.0, 3.0]:
                for cw in ['balanced', None]:
                    cw_label = 'balanced' if cw == 'balanced' else 'unweighted'
                    m_start = time.perf_counter()
                    clf = SVC(kernel='rbf', C=C_val, gamma='scale', class_weight=cw, cache_size=512)
                    clf.fit(lda_combined_tr, y_train)
                    preds = clf.predict(lda_combined_va)
                    acc = float(accuracy_score(y_val, preds))
                    mf1 = float(f1_score(y_val, preds, labels=list(CLASSES), average='macro', zero_division=0))
                    duration = time.perf_counter() - m_start
                    leaderboard.append({
                        'space': f"{space_name} + LDA",
                        'model': f"RBF C={C_val} {cw_label}",
                        'accuracy': acc,
                        'macro_f1': mf1,
                        'duration_s': duration
                    })
                    print(f"  [{space_name:26s} + LDA] RBF C={C_val:3.1f} {cw_label:10s} -> "
                          f"Val Acc: {acc*100:5.2f}% | Macro-F1: {mf1*100:5.2f}% ({duration:.2f}s)", flush=True)

        leaderboard.sort(key=lambda x: (x['macro_f1'], x['accuracy']), reverse=True)
        print(f"\n{'*'*80}", flush=True)
        print("TOP 10 INDIVIDUAL ARCHITECTURES & HYPERPARAMETERS:", flush=True)
        print(f"{'*'*80}", flush=True)
        for rank, item in enumerate(leaderboard[:10], 1):
            print(f"Rank {rank:02d}: Acc = {item['accuracy']*100:5.2f}% | Macro-F1 = {item['macro_f1']*100:5.2f}% | "
                  f"Space: {item['space']} | Model: {item['model']}", flush=True)

    # 5. Build and Fit Calibrated Soft-Voting Ensemble
    with StepTimer("Fitting Calibrated Multi-Backbone Soft-Voting Ensemble"):
        # We use the Dual-Backbone Fused TTA space (4,096-D)
        scaler_ensemble = make_pipeline(StandardScaler(), Normalizer(norm='l2'))
        tr_fused_s = scaler_ensemble.fit_transform(fused_tr_tta)
        va_fused_s = scaler_ensemble.transform(fused_va_tta)

        print("Fitting Model 1: Calibrated RBF SVM (C=2.0, balanced)...", flush=True)
        t_m1 = time.perf_counter()
        clf1 = SVC(kernel='rbf', C=2.0, gamma='scale', class_weight='balanced', probability=True, random_state=42)
        clf1.fit(tr_fused_s, y_train)
        prob1 = clf1.predict_proba(va_fused_s)
        print(f"Model 1 fitted in {time.perf_counter()-t_m1:.2f}s. Individual Val Acc: {accuracy_score(y_val, clf1.classes_[np.argmax(prob1, axis=1)])*100:.2f}%", flush=True)

        print("Fitting Model 2: Calibrated Linear SVM (C=1.0, balanced)...", flush=True)
        t_m2 = time.perf_counter()
        clf2 = SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True, random_state=42)
        clf2.fit(tr_fused_s, y_train)
        prob2 = clf2.predict_proba(va_fused_s)
        print(f"Model 2 fitted in {time.perf_counter()-t_m2:.2f}s. Individual Val Acc: {accuracy_score(y_val, clf2.classes_[np.argmax(prob2, axis=1)])*100:.2f}%", flush=True)

        print("Fitting Model 3: Calibrated RBF SVM (C=3.0, unweighted)...", flush=True)
        t_m3 = time.perf_counter()
        clf3 = SVC(kernel='rbf', C=3.0, gamma='scale', class_weight=None, probability=True, random_state=42)
        clf3.fit(tr_fused_s, y_train)
        prob3 = clf3.predict_proba(va_fused_s)
        print(f"Model 3 fitted in {time.perf_counter()-t_m3:.2f}s. Individual Val Acc: {accuracy_score(y_val, clf3.classes_[np.argmax(prob3, axis=1)])*100:.2f}%", flush=True)

        print("Fitting Model 4: Calibrated Polynomial SVM (degree=2, C=1.5, balanced)...", flush=True)
        t_m4 = time.perf_counter()
        clf4 = SVC(kernel='poly', degree=2, C=1.5, class_weight='balanced', probability=True, random_state=42)
        clf4.fit(tr_fused_s, y_train)
        prob4 = clf4.predict_proba(va_fused_s)
        print(f"Model 4 fitted in {time.perf_counter()-t_m4:.2f}s. Individual Val Acc: {accuracy_score(y_val, clf4.classes_[np.argmax(prob4, axis=1)])*100:.2f}%", flush=True)

        # Optimize Ensemble Weights on Validation Probabilities
        print("\nSearching Optimal Ensemble Mixture Weights...", flush=True)
        best_ens_acc, best_ens_mf1, best_weights = -1.0, -1.0, None
        weight_candidates = [
            [0.40, 0.20, 0.20, 0.20],
            [0.50, 0.25, 0.25, 0.00],
            [0.45, 0.25, 0.15, 0.15],
            [0.35, 0.25, 0.25, 0.15],
            [0.60, 0.20, 0.20, 0.00],
            [0.50, 0.30, 0.10, 0.10],
            [0.25, 0.25, 0.25, 0.25],
        ]

        for w in weight_candidates:
            combined = w[0]*prob1 + w[1]*prob2 + w[2]*prob3 + w[3]*prob4
            preds = np.array([clf1.classes_[idx] for idx in np.argmax(combined, axis=1)])
            acc = float(accuracy_score(y_val, preds))
            mf1 = float(f1_score(y_val, preds, labels=list(CLASSES), average='macro', zero_division=0))
            print(f"  Weights {w} -> Val Acc: {acc*100:.2f}%, Macro-F1: {mf1*100:.2f}%", flush=True)
            if (mf1, acc) > (best_ens_mf1, best_ens_acc):
                best_ens_acc, best_ens_mf1, best_weights = acc, mf1, w

        final_prob = (best_weights[0]*prob1 + best_weights[1]*prob2 +
                      best_weights[2]*prob3 + best_weights[3]*prob4)
        final_preds = np.array([clf1.classes_[idx] for idx in np.argmax(final_prob, axis=1)])
        print(f"\nWinning Ensemble Weights: {best_weights}", flush=True)
        print(f"Final Validation Accuracy: {best_ens_acc*100:.2f}% (Baseline-v1: 68.87%, Baseline-v2: 73.29%)", flush=True)
        print(f"Final Validation Macro-F1: {best_ens_mf1*100:.2f}% (Baseline-v1: 63.36%, Baseline-v2: 69.31%)", flush=True)

    # 6. Detailed Classification Report and Confusion Matrix
    with StepTimer("Evaluating Detailed Diagnostics"):
        rep = classification_report(y_val, final_preds, labels=list(CLASSES), output_dict=True, zero_division=0)
        cm = confusion_matrix(y_val, final_preds, labels=list(CLASSES))

        print("\n" + "="*85, flush=True)
        print("          BASELINE-V3 FINAL VALIDATION CLASSIFICATION REPORT           ", flush=True)
        print("="*85, flush=True)
        print(f"{'Class':15s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}", flush=True)
        print("-"*85, flush=True)
        for c in CLASSES:
            st = rep[c]
            print(f"{c:15s} {st['precision']*100:9.2f}% {st['recall']*100:9.2f}% {st['f1-score']*100:9.2f}% {int(st['support']):10d}", flush=True)
        print("-"*85, flush=True)
        print(f"{'Macro Average':15s} {rep['macro avg']['precision']*100:9.2f}% {rep['macro avg']['recall']*100:9.2f}% {rep['macro avg']['f1-score']*100:9.2f}% {int(rep['macro avg']['support']):10d}", flush=True)
        print(f"{'Weighted Avg':15s} {rep['weighted avg']['precision']*100:9.2f}% {rep['weighted avg']['recall']*100:9.2f}% {rep['weighted avg']['f1-score']*100:9.2f}% {int(rep['weighted avg']['support']):10d}", flush=True)
        print(f"{'Overall Accuracy':15s} {best_ens_acc*100:9.2f}%", flush=True)
        print("="*85, flush=True)

        print("\nCONFUSION MATRIX (Rows: True, Cols: Predicted):", flush=True)
        print(f"{'':15s} " + ' '.join(f"{c[:6]:>6s}" for c in CLASSES), flush=True)
        for c, row in zip(CLASSES, cm):
            print(f"{c:15s} " + ' '.join(f"{val:>6d}" for val in row), flush=True)

    # 7. Publishing Baseline-v3 Artifacts
    with StepTimer("Publishing Baseline-v3 Model Bundle & Reports"):
        out_dir = Path('reports/baseline-v3')
        run_dir = Path('runs/baseline-v3')
        out_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)

        metrics_data = {
            'partition': 'val',
            'images': len(y_val),
            'class_order': list(CLASSES),
            'accuracy': best_ens_acc,
            'macro_f1': best_ens_mf1,
            'baseline_v1_accuracy': 0.6887417218543046,
            'baseline_v1_macro_f1': 0.6335611716880666,
            'baseline_v2_accuracy': 0.7328918322295805,
            'baseline_v2_macro_f1': 0.693099594498998,
            'accuracy_gain_over_v1': best_ens_acc - 0.6887417218543046,
            'macro_f1_gain_over_v1': best_ens_mf1 - 0.6335611716880666,
            'accuracy_gain_over_v2': best_ens_acc - 0.7328918322295805,
            'macro_f1_gain_over_v2': best_ens_mf1 - 0.693099594498998,
            'classification_report': rep,
            'confusion_matrix': cm.tolist(),
            'model_architecture': 'Dual-Backbone (InceptionV3 + ResNet-50 TTA) with Calibrated Soft-Voting SVM Ensemble',
            'ensemble_weights': best_weights,
            'ensemble_members': [
                {'model': 'RBF SVM C=2.0 balanced', 'weight': best_weights[0]},
                {'model': 'Linear SVM C=1.0 balanced', 'weight': best_weights[1]},
                {'model': 'RBF SVM C=3.0 unweighted', 'weight': best_weights[2]},
                {'model': 'Poly SVM degree=2 C=1.5 balanced', 'weight': best_weights[3]}
            ]
        }

        with (out_dir / 'metrics.json').open('w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2)

        with (out_dir / 'confusion_matrix.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['true / predicted', *CLASSES])
            for label, values in zip(CLASSES, cm.tolist()):
                writer.writerow([label, *values])

        with (out_dir / 'validation_predictions.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['path', 'true_label', 'predicted_label'])
            for r, p in zip(val_rows, final_preds):
                writer.writerow([r['path'], r['label'], str(p)])

        joblib.dump({
            'scaler': scaler_ensemble,
            'clf1': clf1,
            'clf2': clf2,
            'clf3': clf3,
            'clf4': clf4,
            'weights': best_weights,
            'classes': list(clf1.classes_)
        }, run_dir / 'ensemble_svm.joblib', compress=3)

        metadata = {
            'schema_version': 3,
            'model': 'max_optimized_dual_backbone_svm_ensemble',
            'backbones': ['InceptionV3', 'ResNet50'],
            'feature_dimensions': 4096,
            'validation_accuracy': best_ens_acc,
            'validation_macro_f1': best_ens_mf1,
            'accuracy_gain_over_v1': best_ens_acc - 0.6887417218543046,
            'macro_f1_gain_over_v1': best_ens_mf1 - 0.6335611716880666,
            'training_images': len(train_rows),
            'validation_images': len(val_rows),
            'class_order': list(CLASSES),
            'generated_at_utc': datetime.now(timezone.utc).isoformat()
        }

        with (run_dir / 'metadata.json').open('w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        with (out_dir / 'run.json').open('w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

    total_time = time.perf_counter() - overall_start
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] PIPELINE COMPLETE! Total Execution Time: {total_time:.2f}s ({total_time/60:.2f} min)", flush=True)


if __name__ == '__main__':
    main()
