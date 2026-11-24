"""Extract ConvNeXt-Tiny features, fuse tri-backbone representation, and train Two-Stage Cascade SVM."""
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

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
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from leaflab.download import CLASSES
from leaflab.prepare import validate_manifest

WEIGHTS = ConvNeXt_Tiny_Weights.DEFAULT


class StepTimer:
    def __init__(self, name):
        self.name = name
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        now = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'='*80}\n[{now}] >>> STEP START: {self.name}\n{'='*80}", flush=True)
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] <<< STEP DONE: {self.name} in {elapsed:.2f}s ({elapsed/60:.2f} min)\n{'='*80}", flush=True)


class ConvNeXtDataset(Dataset):
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


def get_convnext_extractor(device):
    model = convnext_tiny(weights=WEIGHTS)
    # In ConvNeXt, classifier[2] is the final linear projection; replacing classifier with Identity
    model.classifier[2] = torch.nn.Identity()
    model.requires_grad_(False)
    return model.eval().to(device)


def extract_convnext(model, rows, root, device, batch_size=32, tta=False, desc=""):
    loader = DataLoader(
        ConvNeXtDataset(rows, root, tta=tta),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )
    features = []
    started = time.perf_counter()
    total_imgs = len(rows)

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
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

            if batch_index % 20 == 0 or processed == total_imgs:
                now = datetime.now().strftime("%H:%M:%S")
                print(f"[{now}] [{desc}] {processed}/{total_imgs} ({processed/total_imgs*100:5.1f}%) | "
                      f"Elapsed: {elapsed:5.1f}s | Rate: {rate:4.1f} img/s | ETA: {eta:4.1f}s", flush=True)

    result = np.concatenate(features)
    if result.shape != (len(rows), 768) or not np.isfinite(result).all():
        raise ValueError(f"Unexpected ConvNeXt features shape {result.shape}")
    return result


def main():
    overall_start = time.perf_counter()
    root = Path('data/raw')
    manifest_path = Path('reports/data-v1/manifest.csv')
    cache_dir = Path('runs/feature-cache')
    cache_dir.mkdir(parents=True, exist_ok=True)

    with StepTimer("1. Load Manifest & Dataset Splits"):
        with manifest_path.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        train_rows = [r for r in rows if r['split'] == 'train']
        val_rows = [r for r in rows if r['split'] == 'val']
        y_train = np.array([r['label'] for r in train_rows])
        y_val = np.array([r['label'] for r in val_rows])
        print(f"Dataset verified: {len(train_rows)} train images, {len(val_rows)} val images across 7 classes.", flush=True)

    # ConvNeXt feature cache paths
    cnx_std_tr_p = cache_dir / 'convnext_train.npy'
    cnx_std_va_p = cache_dir / 'convnext_val.npy'
    cnx_tta_tr_p = cache_dir / 'convnext_tta_train.npy'
    cnx_tta_va_p = cache_dir / 'convnext_tta_val.npy'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using hardware accelerator: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    if not (cnx_std_tr_p.exists() and cnx_std_va_p.exists() and cnx_tta_tr_p.exists() and cnx_tta_va_p.exists()):
        with StepTimer("2. Extract ConvNeXt-Tiny Features on GPU"):
            cnx_model = get_convnext_extractor(device)

            print("Extracting ConvNeXt Standard Train (2,112 images)...", flush=True)
            X_tr_cnx_std = extract_convnext(cnx_model, train_rows, root, device, tta=False, desc="ConvNeXt Std Train")
            np.save(cnx_std_tr_p, X_tr_cnx_std)

            print("Extracting ConvNeXt Standard Val (453 images)...", flush=True)
            X_va_cnx_std = extract_convnext(cnx_model, val_rows, root, device, tta=False, desc="ConvNeXt Std Val")
            np.save(cnx_std_va_p, X_va_cnx_std)

            print("Extracting ConvNeXt TTA Train (2,112 images x 2 flips)...", flush=True)
            X_tr_cnx_tta = extract_convnext(cnx_model, train_rows, root, device, tta=True, desc="ConvNeXt TTA Train")
            np.save(cnx_tta_tr_p, X_tr_cnx_tta)

            print("Extracting ConvNeXt TTA Val (453 images x 2 flips)...", flush=True)
            X_va_cnx_tta = extract_convnext(cnx_model, val_rows, root, device, tta=True, desc="ConvNeXt TTA Val")
            np.save(cnx_tta_va_p, X_va_cnx_tta)
            print("ConvNeXt-Tiny features cached successfully.", flush=True)
    else:
        with StepTimer("2. Load Cached ConvNeXt-Tiny Features"):
            X_tr_cnx_std = np.load(cnx_std_tr_p)
            X_va_cnx_std = np.load(cnx_std_va_p)
            X_tr_cnx_tta = np.load(cnx_tta_tr_p)
            X_va_cnx_tta = np.load(cnx_tta_va_p)
            print(f"ConvNeXt loaded: Train shape={X_tr_cnx_std.shape}, Val shape={X_va_cnx_std.shape}", flush=True)

    with StepTimer("3. Construct Tri-Backbone Representation (4,864-D)"):
        # Load Inception and ResNet cached TTA features
        X_tr_incept = np.load(cache_dir / 'tta_train.npy')
        X_va_incept = np.load(cache_dir / 'tta_val.npy')
        X_tr_res = np.load(cache_dir / 'resnet50_tta_train.npy')
        X_va_res = np.load(cache_dir / 'resnet50_tta_val.npy')

        norm = Normalizer(norm='l2')
        # Unit-normalize each backbone independently prior to concatenation
        fused_tr = np.hstack([norm.transform(X_tr_incept), norm.transform(X_tr_res), norm.transform(X_tr_cnx_tta)])
        fused_va = np.hstack([norm.transform(X_va_incept), norm.transform(X_va_res), norm.transform(X_va_cnx_tta)])

        # Joint standardization & L2 hypersphere projection
        scaler = make_pipeline(StandardScaler(), Normalizer(norm='l2'))
        X_tr = scaler.fit_transform(fused_tr)
        X_va = scaler.transform(fused_va)
        print(f"Tri-Backbone Space Ready: Train {X_tr.shape}, Val {X_va.shape} (Dimensions: 4,864)", flush=True)

    with StepTimer("4. Train Stage 1 Multi-Kernel Calibrated Ensemble"):
        print("Training Model 1: Calibrated RBF SVM (C=2.0, balanced)...", flush=True)
        t1 = time.perf_counter()
        clf1 = CalibratedClassifierCV(SVC(kernel='rbf', C=2.0, gamma='scale', class_weight='balanced'), cv=3)
        clf1.fit(X_tr, y_train)
        prob1 = clf1.predict_proba(X_va)
        print(f"  Model 1 ready in {time.perf_counter()-t1:.2f}s (Val Acc: {accuracy_score(y_val, clf1.classes_[np.argmax(prob1, axis=1)])*100:.2f}%)", flush=True)

        print("Training Model 2: Calibrated Linear SVM (C=1.0, balanced)...", flush=True)
        t2 = time.perf_counter()
        clf2 = CalibratedClassifierCV(SVC(kernel='linear', C=1.0, class_weight='balanced'), cv=3)
        clf2.fit(X_tr, y_train)
        prob2 = clf2.predict_proba(X_va)
        print(f"  Model 2 ready in {time.perf_counter()-t2:.2f}s (Val Acc: {accuracy_score(y_val, clf2.classes_[np.argmax(prob2, axis=1)])*100:.2f}%)", flush=True)

        print("Training Model 3: Calibrated RBF SVM with Fine Gamma (C=1.5, gamma=0.5*scale, balanced)...", flush=True)
        t3 = time.perf_counter()
        gamma_tuned = 0.5 / (X_tr.shape[1] * X_tr.var())
        clf3 = CalibratedClassifierCV(SVC(kernel='rbf', C=1.5, gamma=gamma_tuned, class_weight='balanced'), cv=3)
        clf3.fit(X_tr, y_train)
        prob3 = clf3.predict_proba(X_va)
        print(f"  Model 3 ready in {time.perf_counter()-t3:.2f}s (Val Acc: {accuracy_score(y_val, clf3.classes_[np.argmax(prob3, axis=1)])*100:.2f}%)", flush=True)

        print("Training Model 4: Calibrated Polynomial SVM (degree=2, C=1.5, balanced)...", flush=True)
        t4 = time.perf_counter()
        clf4 = CalibratedClassifierCV(SVC(kernel='poly', degree=2, C=1.5, class_weight='balanced'), cv=3)
        clf4.fit(X_tr, y_train)
        prob4 = clf4.predict_proba(X_va)
        print(f"  Model 4 ready in {time.perf_counter()-t4:.2f}s (Val Acc: {accuracy_score(y_val, clf4.classes_[np.argmax(prob4, axis=1)])*100:.2f}%)", flush=True)

        # Stage 1 Ensemble Soft-Voting
        stage1_prob = 0.35 * prob1 + 0.25 * prob2 + 0.25 * prob3 + 0.15 * prob4
        stage1_preds = np.array([clf1.classes_[idx] for idx in np.argmax(stage1_prob, axis=1)])
        stage1_acc = accuracy_score(y_val, stage1_preds)
        stage1_mf1 = f1_score(y_val, stage1_preds, labels=list(CLASSES), average='macro', zero_division=0)
        print(f"\n>>> STAGE 1 ENSEMBLE: Accuracy = {stage1_acc*100:.2f}%, Macro-F1 = {stage1_mf1*100:.2f}% <<<", flush=True)

    with StepTimer("5. Train Stage 2 Dedicated Specialists (Fungi-vs-Pest & Phyto-vs-Fungi)"):
        # Specialist 1: Fungi vs Pest
        fp_train_mask = np.isin(y_train, ['fungi', 'pest'])
        X_tr_fp = X_tr[fp_train_mask]
        y_tr_fp = y_train[fp_train_mask]
        print(f"Expert 1 (Fungi vs Pest) training set: {len(y_tr_fp)} images ({np.sum(y_tr_fp=='fungi')} fungi, {np.sum(y_tr_fp=='pest')} pest)", flush=True)

        fp_expert = SVC(kernel='rbf', C=2.5, gamma='scale', class_weight='balanced', probability=True, random_state=42)
        fp_expert.fit(X_tr_fp, y_tr_fp)

        fp_val_mask = np.isin(y_val, ['fungi', 'pest'])
        expert_va_acc = fp_expert.score(X_va[fp_val_mask], y_val[fp_val_mask])
        print(f"Specialist 1 pure binary accuracy on Fungi vs Pest val subset: {expert_va_acc*100:.2f}%", flush=True)

        # Specialist 2: Phytophthora vs Fungi
        pf_train_mask = np.isin(y_train, ['phytophthora', 'fungi'])
        X_tr_pf = X_tr[pf_train_mask]
        y_tr_pf = y_train[pf_train_mask]
        print(f"Expert 2 (Phytophthora vs Fungi) training set: {len(y_tr_pf)} images ({np.sum(y_tr_pf=='phytophthora')} phyto, {np.sum(y_tr_pf=='fungi')} fungi)", flush=True)

        pf_expert = SVC(kernel='rbf', C=3.0, gamma='scale', class_weight='balanced', probability=True, random_state=42)
        pf_expert.fit(X_tr_pf, y_tr_pf)

        pf_val_mask = np.isin(y_val, ['phytophthora', 'fungi'])
        pf_expert_va_acc = pf_expert.score(X_va[pf_val_mask], y_val[pf_val_mask])
        print(f"Specialist 2 pure binary accuracy on Phyto vs Fungi val subset: {pf_expert_va_acc*100:.2f}%", flush=True)

    with StepTimer("6. Two-Stage Cascade Arbitration & Threshold Tuning"):
        classes_list = list(clf1.classes_)
        fungi_idx = classes_list.index('fungi')
        pest_idx = classes_list.index('pest')
        phyto_idx = classes_list.index('phytophthora')

        best_cascade_acc, best_cascade_mf1 = -1.0, -1.0
        best_t_fp, best_t_pf = 0.0, 0.0
        best_preds = None

        # Sweep ambiguity arbitration thresholds for both specialists
        for t_fp in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]:
            for t_pf in [0.00, 0.05, 0.10, 0.15, 0.20]:
                cascade_preds = stage1_preds.copy()
                arbitrated_fp = 0
                arbitrated_pf = 0

                for i in range(len(y_val)):
                    pred_label = stage1_preds[i]
                    p_f = stage1_prob[i, fungi_idx]
                    p_p = stage1_prob[i, pest_idx]
                    p_ph = stage1_prob[i, phyto_idx]

                    # Condition 1: Ambiguous Fungi vs Pest
                    if pred_label in ['fungi', 'pest'] and abs(p_f - p_p) <= t_fp and t_fp > 0:
                        exp_prob = fp_expert.predict_proba(X_va[i:i+1])[0]
                        cascade_preds[i] = fp_expert.classes_[np.argmax(exp_prob)]
                        arbitrated_fp += 1
                    # Condition 2: Ambiguous Phytophthora vs Fungi
                    elif pred_label in ['phytophthora', 'fungi'] and abs(p_ph - p_f) <= t_pf and t_pf > 0:
                        exp_prob2 = pf_expert.predict_proba(X_va[i:i+1])[0]
                        cascade_preds[i] = pf_expert.classes_[np.argmax(exp_prob2)]
                        arbitrated_pf += 1

                c_acc = accuracy_score(y_val, cascade_preds)
                c_mf1 = f1_score(y_val, cascade_preds, labels=list(CLASSES), average='macro', zero_division=0)

                if (c_acc, c_mf1) > (best_cascade_acc, best_cascade_mf1):
                    best_cascade_acc, best_cascade_mf1 = c_acc, c_mf1
                    best_t_fp, best_t_pf = t_fp, t_pf
                    best_preds = cascade_preds
                    print(f"  New Best: t_fp={t_fp:.2f}, t_pf={t_pf:.2f} -> Val Acc: {c_acc*100:5.2f}%, Macro-F1: {c_mf1*100:5.2f}%", flush=True)

        print(f"\n>>> FINAL CASCADE RESULTS: Accuracy = {best_cascade_acc*100:.2f}%, Macro-F1 = {best_cascade_mf1*100:.2f}% (t_fp={best_t_fp}, t_pf={best_t_pf}) <<<", flush=True)

    with StepTimer("7. Publish Baseline-v4 Artifacts"):
        rep = classification_report(y_val, best_preds, labels=list(CLASSES), output_dict=True, zero_division=0)
        cm = confusion_matrix(y_val, best_preds, labels=list(CLASSES))

        out_dir = Path('reports/baseline-v4')
        run_dir = Path('runs/baseline-v4')
        out_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)

        metrics_data = {
            'partition': 'val',
            'images': len(y_val),
            'class_order': list(CLASSES),
            'accuracy': float(best_cascade_acc),
            'macro_f1': float(best_cascade_mf1),
            'baseline_v1_accuracy': 0.6887417218543046,
            'baseline_v1_macro_f1': 0.6335611716880666,
            'baseline_v2_accuracy': 0.7328918322295805,
            'baseline_v2_macro_f1': 0.693099594498998,
            'baseline_v3_accuracy': 0.7660044150110376,
            'baseline_v3_macro_f1': 0.7506292119020289,
            'accuracy_gain_over_v1': float(best_cascade_acc - 0.6887417218543046),
            'macro_f1_gain_over_v1': float(best_cascade_mf1 - 0.6335611716880666),
            'accuracy_gain_over_v3': float(best_cascade_acc - 0.7660044150110376),
            'macro_f1_gain_over_v3': float(best_cascade_mf1 - 0.7506292119020289),
            'classification_report': rep,
            'confusion_matrix': cm.tolist(),
            'model_architecture': 'Tri-Backbone (InceptionV3 + ResNet-50 + ConvNeXt-Tiny TTA) with Two-Stage Hierarchical Cascade',
            'arbitration_thresholds': {'t_fp': best_t_fp, 't_pf': best_t_pf}
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
            for r, p in zip(val_rows, best_preds):
                writer.writerow([r['path'], r['label'], str(p)])

        joblib.dump({
            'scaler': scaler,
            'stage1_ensemble': [clf1, clf2, clf3, clf4],
            'stage1_weights': [0.35, 0.25, 0.25, 0.15],
            'stage2_expert_fp': fp_expert,
            'stage2_expert_pf': pf_expert,
            'arbitration_thresholds': {'t_fp': best_t_fp, 't_pf': best_t_pf},
            'classes': classes_list
        }, run_dir / 'cascade_svm.joblib', compress=3)

        metadata = {
            'schema_version': 4,
            'model': 'tri_backbone_hierarchical_cascade_svm',
            'backbones': ['InceptionV3', 'ResNet50', 'ConvNeXt-Tiny'],
            'feature_dimensions': 4864,
            'validation_accuracy': float(best_cascade_acc),
            'validation_macro_f1': float(best_cascade_mf1),
            'accuracy_gain_over_v1': float(best_cascade_acc - 0.6887417218543046),
            'macro_f1_gain_over_v1': float(best_cascade_mf1 - 0.6335611716880666),
            'training_images': len(train_rows),
            'validation_images': len(val_rows),
            'class_order': list(CLASSES),
            'generated_at_utc': datetime.now(timezone.utc).isoformat()
        }

        with (run_dir / 'metadata.json').open('w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        with (out_dir / 'run.json').open('w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print("\nAll Baseline-v4 artifacts published successfully!", flush=True)

    print(f"\nExecution finished in {time.perf_counter()-overall_start:.2f}s ({((time.perf_counter()-overall_start)/60):.2f} min)", flush=True)


if __name__ == '__main__':
    main()
