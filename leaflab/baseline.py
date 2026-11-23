"""Frozen ImageNet InceptionV3 features and training-only SVM fitting."""
import csv
import hashlib
import json
import os
import platform
import tempfile
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Inception_V3_Weights, inception_v3

from .download import CLASSES, sha256
from .prepare import validate_manifest

WEIGHTS = Inception_V3_Weights.IMAGENET1K_V1
PREPROCESSING = {
    'weights': 'Inception_V3_Weights.IMAGENET1K_V1',
    'exif_transpose': True,
    'color': 'RGB',
    'resize_shorter_edge': 342,
    'center_crop': 299,
    'interpolation': 'bilinear',
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'inception_transform_input': True,
    'feature_layer': 'global_average_pool_before_fc',
    'feature_dimensions': 2048,
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def select_device(requested):
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but PyTorch cannot access the GPU')
    return torch.device(requested)


def configure_torch():
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def environment_report():
    result = {
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(),
        'platform': platform.platform(),
        'packages': {name: version(name) for name in
                     ['torch', 'torchvision', 'numpy', 'scikit-learn', 'Pillow', 'joblib']},
        'cuda_available': torch.cuda.is_available(),
        'torch_cuda_runtime': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
    }
    if result['cuda_available']:
        properties = torch.cuda.get_device_properties(0)
        value = torch.ones((32, 32), device='cuda')
        torch.cuda.synchronize()
        result.update(gpu=properties.name, gpu_memory_bytes=properties.total_memory,
                      cuda_matrix_multiply_passed=bool(torch.all(value @ value == 32).item()))
    return result


def check_environment(out):
    configure_torch()
    result = environment_report()
    write_json(out, result)
    print(json.dumps(result, indent=2), flush=True)


class ManifestImages(Dataset):
    def __init__(self, rows, root):
        self.rows = rows
        self.root = Path(root)
        self.transform = WEIGHTS.transforms()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.root / self.rows[index]['path']) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            return self.transform(image)


def feature_extractor(device, state_path=None):
    if state_path is None:
        model = inception_v3(weights=WEIGHTS)
        model.AuxLogits = None
        model.aux_logits = False
    else:
        model = inception_v3(weights=None, aux_logits=False, transform_input=True,
                             init_weights=False)
    model.fc = torch.nn.Identity()
    if state_path is not None:
        model.load_state_dict(torch.load(state_path, map_location='cpu', weights_only=True))
    model.requires_grad_(False)
    return model.eval().to(device)


def extract_features(model, rows, root, device, batch_size, workers):
    loader = DataLoader(ManifestImages(rows, root), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=device.type == 'cuda')
    features = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            output = model(batch.to(device, non_blocking=device.type == 'cuda'))
            features.append(output.cpu().numpy())
            if batch_index % 10 == 0 or batch_index + 1 == len(loader):
                count = min((batch_index + 1) * batch_size, len(rows))
                print(f'Features {count}/{len(rows)} ({time.perf_counter()-started:.1f}s)', flush=True)
    result = np.concatenate(features)
    if result.shape != (len(rows), 2048) or not np.isfinite(result).all():
        raise ValueError('Unexpected or non-finite InceptionV3 features')
    return result


def feature_cache_key(rows, manifest_hash, device):
    settings = {'manifest_sha256': manifest_hash, 'preprocessing': PREPROCESSING,
                'torch': torch.__version__, 'torchvision': version('torchvision'),
                'device': str(device), 'dtype': 'float32',
                'paths': [row['path'] for row in rows]}
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def cached_features(model, rows, root, manifest_hash, cache, device, batch_size, workers):
    key = feature_cache_key(rows, manifest_hash, device)
    path = Path(cache) / f'{key}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            result = saved['features']
            if (saved['key'].item() != key or result.shape != (len(rows), 2048)
                    or not np.isfinite(result).all()):
                raise ValueError(f'Invalid feature cache: {path}')
        print(f'Reused verified-input feature cache for {len(rows)} images', flush=True)
        return result
    result = extract_features(model, rows, root, device, batch_size, workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, features=result, key=np.array(key))
    temporary.replace(path)
    return result


def fit_candidates(train_features, train_labels, val_features, val_labels):
    """Select by validation macro-F1, then accuracy, retaining the first exact tie."""
    results = []
    best_model, best_predictions, best_score = None, None, (-1.0, -1.0)
    for kernel in ['linear', 'rbf']:
        for penalty in [0.1, 1.0, 10.0, 100.0]:
            parameters = {'kernel': kernel, 'C': penalty, 'gamma': 'scale',
                          'class_weight': None}
            model = make_pipeline(StandardScaler(), SVC(**parameters, cache_size=512))
            started = time.perf_counter()
            model.fit(train_features, train_labels)
            predictions = model.predict(val_features)
            score = (f1_score(val_labels, predictions, labels=list(CLASSES),
                              average='macro', zero_division=0),
                     accuracy_score(val_labels, predictions))
            result = {'parameters': parameters, 'validation_macro_f1': float(score[0]),
                      'validation_accuracy': float(score[1]),
                      'fit_and_predict_seconds': time.perf_counter() - started}
            results.append(result)
            print(f'SVM {kernel} C={penalty:g}: val accuracy={score[1]:.4f}, '
                  f'macro-F1={score[0]:.4f}', flush=True)
            if score > best_score:
                best_model, best_predictions, best_score = model, predictions, score
    return best_model, best_predictions, results


def write_validation_reports(directory, rows, predictions, candidates):
    labels = [row['label'] for row in rows]
    report = classification_report(labels, predictions, labels=list(CLASSES),
                                   output_dict=True, zero_division=0)
    metrics = {'partition': 'val', 'images': len(rows), 'class_order': list(CLASSES),
               'accuracy': float(accuracy_score(labels, predictions)),
               'macro_f1': float(f1_score(labels, predictions, labels=list(CLASSES),
                                          average='macro', zero_division=0)),
               'classification_report': report,
               'confusion_matrix': confusion_matrix(labels, predictions,
                                                     labels=list(CLASSES)).tolist()}
    write_json(directory / 'metrics.json', metrics)
    write_json(directory / 'candidates.json', candidates)
    with (directory / 'validation_predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['path', 'true_label', 'predicted_label'])
        writer.writerows((row['path'], row['label'], str(predicted))
                         for row, predicted in zip(rows, predictions, strict=True))
    with (directory / 'confusion_matrix.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['true / predicted', *CLASSES])
        writer.writerows([label, *values] for label, values in
                         zip(CLASSES, metrics['confusion_matrix'], strict=True))
    return metrics


def run_baseline(args):
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError('Batch size must be positive and workers non-negative')
    out, report = Path(args.out), Path(args.report)
    if out.resolve() == report.resolve() or out.exists() or report.exists():
        raise FileExistsError('Choose separate, new output/report directories for this run')
    configure_torch()
    device = select_device(args.device)
    environment = environment_report()
    print(f'Checking dataset hashes before feature extraction on {device}', flush=True)
    rows = validate_manifest(args.manifest, args.data)
    train_rows = [row for row in rows if row['split'] == 'train']
    val_rows = [row for row in rows if row['split'] == 'val']
    manifest_hash = sha256(args.manifest)
    torch.hub.set_dir(str(Path(args.cache).parent / 'torch-hub'))
    model = feature_extractor(device)
    train_features = cached_features(model, train_rows, args.data, manifest_hash, args.cache,
                                    device, args.batch_size, args.workers)
    val_features = cached_features(model, val_rows, args.data, manifest_hash, args.cache,
                                  device, args.batch_size, args.workers)
    classifier, predictions, candidates = fit_candidates(
        train_features, [r['label'] for r in train_rows],
        val_features, [r['label'] for r in val_rows])
    out.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    # A run is usable only after its complete bundle is published from the staging directory.
    with tempfile.TemporaryDirectory(prefix='.baseline-', dir=out.parent) as staging:
        bundle = Path(staging) / 'bundle'
        bundle.mkdir()
        torch.save(model.cpu().state_dict(), bundle / 'feature_extractor.pt')
        joblib.dump(classifier, bundle / 'svm.joblib', compress=3)
        selected = classifier.named_steps['svc'].get_params()
        metadata = {
            'schema_version': 1, 'model': 'frozen_inception_v3_plus_svm',
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'manifest_sha256': manifest_hash, 'seed': 42,
            'training_images': len(train_rows), 'validation_images': len(val_rows),
            'test_images_used_for_fitting_or_selection': 0,
            'preprocessing': PREPROCESSING, 'class_order': list(CLASSES),
            'svm': {key: selected[key] for key in ['kernel', 'C', 'gamma', 'class_weight']},
            'selection_rule': 'highest validation macro-F1, then accuracy, then first candidate',
            'candidate_count': len(candidates), 'environment': environment,
            'feature_extractor_sha256': sha256(bundle / 'feature_extractor.pt'),
            'svm_sha256': sha256(bundle / 'svm.joblib'),
            'complete_bundle_bytes_excluding_metadata': sum(p.stat().st_size for p in bundle.iterdir()),
        }
        write_json(bundle / 'metadata.json', metadata)
        with tempfile.TemporaryDirectory(prefix='.baseline-report-', dir=report.parent) as temp_report:
            report_bundle = Path(temp_report) / 'report'
            report_bundle.mkdir()
            metrics = write_validation_reports(report_bundle, val_rows, predictions, candidates)
            write_json(report_bundle / 'run.json', metadata)
            bundle.rename(out)
            report_bundle.rename(report)
    print(json.dumps({'saved_model': str(out), 'report': str(report),
                      'validation_accuracy': metrics['accuracy'],
                      'validation_macro_f1': metrics['macro_f1']}, indent=2), flush=True)


def predict_image(model_path, image_path, requested_device='auto'):
    configure_torch()
    root = Path(model_path)
    metadata = json.loads((root / 'metadata.json').read_text(encoding='utf-8'))
    if metadata['schema_version'] != 1 or metadata['preprocessing'] != PREPROCESSING:
        raise ValueError('Unsupported model bundle or preprocessing configuration')
    for filename, field in [('feature_extractor.pt', 'feature_extractor_sha256'),
                            ('svm.joblib', 'svm_sha256')]:
        if sha256(root / filename) != metadata[field]:
            raise ValueError(f'Model artifact has changed: {filename}')
    device = select_device(requested_device)
    model = feature_extractor(device, root / 'feature_extractor.pt')
    # Only load joblib bundles produced by a trusted local training run.
    classifier = joblib.load(root / 'svm.joblib')
    with Image.open(image_path) as image:
        tensor = WEIGHTS.transforms()(ImageOps.exif_transpose(image).convert('RGB'))
    with torch.inference_mode():
        features = model(tensor.unsqueeze(0).to(device)).cpu().numpy()
    category = str(classifier.predict(features)[0])
    if category not in metadata['class_order']:
        raise ValueError('Classifier returned an unknown category')
    result = {'image': str(image_path), 'predicted_category': category}
    print(json.dumps(result, indent=2), flush=True)
    return result
