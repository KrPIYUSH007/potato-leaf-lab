"""Versioned frozen-feature SVM experiment with group-aware calibration and inference."""
import argparse
import hashlib
import json
import shutil
import tempfile
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from .baseline import (configure_torch, environment_report, select_device, write_json,
                       write_validation_reports)
import joblib
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (ConvNeXt_Tiny_Weights, EfficientNet_B0_Weights,
                                Inception_V3_Weights, ResNet50_Weights, convnext_tiny,
                                efficientnet_b0, inception_v3, resnet50)

from .download import CLASSES, sha256
from .prepare import validate_manifest

BACKBONES = {
    'inception': (inception_v3, Inception_V3_Weights.IMAGENET1K_V1, 2048),
    'resnet50': (resnet50, ResNet50_Weights.IMAGENET1K_V2, 2048),
    'convnext': (convnext_tiny, ConvNeXt_Tiny_Weights.IMAGENET1K_V1, 768),
    'efficientnet': (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1, 1280),
}
REPRESENTATIONS = {
    'convnext': ['convnext'],
    'tri': ['inception', 'resnet50', 'convnext'],
    'quad': ['inception', 'resnet50', 'convnext', 'efficientnet'],
}


def build_extractor(name, state=None):
    builder, weights, _ = BACKBONES[name]
    options = {'weights': weights if state is None else None, 'progress': False}
    if name == 'inception' and state is not None:
        options.update(aux_logits=False, transform_input=True, init_weights=False)
    model = builder(**options)
    if name == 'inception':
        model.AuxLogits = None
        model.aux_logits = False
    if name in ('inception', 'resnet50'):
        model.fc = torch.nn.Identity()
    elif name == 'convnext':
        model.classifier[-1] = torch.nn.Identity()
    else:
        model.classifier = torch.nn.Identity()
    if state is not None:
        model.load_state_dict(torch.load(state, map_location='cpu', weights_only=True))
    return model.eval().requires_grad_(False)


class FlipImages(Dataset):
    def __init__(self, rows, root, name):
        self.rows, self.root = rows, Path(root)
        self.transform = BACKBONES[name][1].transforms()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.root / self.rows[index]['path']) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
            return torch.stack([self.transform(image), self.transform(ImageOps.mirror(image))])


def cache_settings(name, rows, manifest_hash):
    return {'schema': 5, 'backbone': name, 'weights': str(BACKBONES[name][1]),
            'transform': str(BACKBONES[name][1].transforms()),
            'tta': 'original_and_horizontal_flip_mean', 'dtype': 'float32',
            'torch': str(torch.__version__), 'torchvision': version('torchvision'),
            'manifest_sha256': manifest_hash,
            'rows': [(r['path'], r['file_sha256']) for r in rows]}


def get_features(name, rows, args, manifest_hash, device, state_path):
    settings = cache_settings(name, rows, manifest_hash)
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    path = Path(args.cache) / f'{name}-{key}.npz'
    model = build_extractor(name)
    if not state_path.exists():
        temporary = state_path.with_suffix('.tmp')
        torch.save(model.state_dict(), temporary)
        temporary.replace(state_path)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if saved['key'].item() != key:
                raise ValueError('Feature cache provenance mismatch')
            features = saved['features']
    else:
        model.to(device)
        loader = DataLoader(FlipImages(rows, args.data, name), batch_size=args.batch_size,
                            num_workers=args.workers, shuffle=False,
                            pin_memory=device.type == 'cuda')
        chunks = []
        started = time.perf_counter()
        with torch.inference_mode():
            for index, images in enumerate(loader):
                batch = images.shape[0]
                output = model(images.flatten(0, 1).to(device))
                chunks.append(output.reshape(batch, 2, -1).mean(1).cpu().numpy())
                if index % 30 == 0 or index + 1 == len(loader):
                    print(f'{name}: {min((index+1)*args.batch_size,len(rows))}/{len(rows)} '
                          f'images, {time.perf_counter()-started:.1f}s', flush=True)
        features = np.concatenate(chunks)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, key=np.array(key), features=features)
        temporary.replace(path)
    if features.shape != (len(rows), BACKBONES[name][2]) or not np.isfinite(features).all():
        raise ValueError(f'Invalid {name} features')
    del model
    return features


def fuse(features, representation):
    return np.hstack([Normalizer().transform(features[name])
                      for name in REPRESENTATIONS[representation]])


def svm_pipeline(parameters):
    return make_pipeline(StandardScaler(), Normalizer(), SVC(**parameters, cache_size=512))


def group_folds(labels, groups):
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    folds = list(splitter.split(np.zeros(len(labels)), labels, groups))
    for train, validation in folds:
        if set(groups[train]) & set(groups[validation]):
            raise ValueError('Calibration group overlap')
        if set(labels[train]) != set(labels) or set(labels[validation]) != set(labels):
            raise ValueError('A calibration fold is missing a category')
    return folds


def metric_pair(labels, predictions):
    return (float(f1_score(labels, predictions, labels=list(CLASSES),
                           average='macro', zero_division=0)),
            float(accuracy_score(labels, predictions)))


def voted_probabilities(members, features, weights):
    probability = np.zeros((len(next(iter(features.values()))), len(CLASSES)))
    for member, weight in zip(members, weights, strict=True):
        model = member['model']
        values = model.predict_proba(fuse(features, member['representation']))
        columns = [list(model.classes_).index(label) for label in CLASSES]
        probability += weight * values[:, columns]
    return probability


def apply_specialists(probability, expert_predictions, thresholds):
    predicted = np.array(CLASSES)[probability.argmax(axis=1)]
    original = predicted.copy()
    for index, (left, right) in enumerate([('fungi', 'pest'), ('phytophthora', 'fungi')]):
        gap = abs(probability[:, CLASSES.index(left)] - probability[:, CLASSES.index(right)])
        mask = np.isin(original, [left, right]) & (gap <= thresholds[index])
        if thresholds[index] == 0:
            mask[:] = False
        if index == 0:
            first_mask = mask.copy()
        else:
            mask &= ~first_mask
        predicted[mask] = expert_predictions[index][mask]
    return predicted


def train(args):
    configure_torch()
    out, report = Path(args.out), Path(args.report)
    if out.exists() or report.exists() or out.resolve() == report.resolve():
        raise FileExistsError('Choose new, distinct run and report directories')
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError('Invalid batch size or worker count')
    work = Path(args.cache)
    work.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(work.parent / 'torch-hub'))
    device = select_device(args.device)
    print('Verifying manifest and image hashes', flush=True)
    rows = validate_manifest(args.manifest, args.data)
    train_rows = [r for r in rows if r['split'] == 'train']
    val_rows = [r for r in rows if r['split'] == 'val']
    labels = np.array([r['label'] for r in train_rows])
    val_labels = np.array([r['label'] for r in val_rows])
    groups = np.array([r['group'] for r in train_rows])
    manifest_hash = sha256(args.manifest)
    plan = {'representations': REPRESENTATIONS, 'C': [0.5, 2.0, 8.0],
            'class_weight': [None, 'balanced'], 'kernel': 'rbf', 'gamma': 'scale',
            'vote_weights': [[1/3, 1/3, 1/3], [0.6, 0.2, 0.2],
                             [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]],
            'specialist_thresholds': [0.0, 0.1, 0.2],
            'selection': 'macro_f1_then_accuracy_then_first_candidate',
            'test_images_used': 0}
    write_json(work / 'plan.json', plan)
    features, states = {}, {}
    for name in BACKBONES:
        states[name] = work / f'{name}-weights.pt'
        features[name] = get_features(name, train_rows + val_rows, args,
                                      manifest_hash, device, states[name])
    training = {name: values[:len(train_rows)] for name, values in features.items()}
    validation = {name: values[len(train_rows):] for name, values in features.items()}
    candidates, top_parameters = [], {}
    best_score, best_predictions, best_bundle = (-1.0, -1.0), None, None
    folds = group_folds(labels, groups)
    members = []
    for representation in REPRESENTATIONS:
        x_train, x_val = fuse(training, representation), fuse(validation, representation)
        top_score = (-1.0, -1.0)
        for penalty in plan['C']:
            for balancing in plan['class_weight']:
                parameters = {'kernel': 'rbf', 'C': penalty, 'gamma': 'scale',
                              'class_weight': balancing}
                model = svm_pipeline(parameters).fit(x_train, labels)
                predictions = model.predict(x_val)
                score = metric_pair(val_labels, predictions)
                candidates.append({'kind': 'single', 'representation': representation,
                                   'parameters': parameters, 'macro_f1': score[0], 'accuracy': score[1]})
                print(f'{representation} C={penalty} balance={balancing}: '
                      f'accuracy={score[1]:.4f}, macro-F1={score[0]:.4f}', flush=True)
                if score > top_score:
                    top_score, top_parameters[representation] = score, parameters
                if score > best_score:
                    best_score, best_predictions = score, predictions
                    best_bundle = {'mode': 'single', 'members': [
                        {'representation': representation, 'model': model}]}
        calibrated = CalibratedClassifierCV(svm_pipeline(top_parameters[representation]), cv=folds)
        calibrated.fit(x_train, labels)
        members.append({'representation': representation, 'model': calibrated})
        print(f'{representation}: completed group-aware probability calibration', flush=True)
    best_vote_score = (-1.0, -1.0)
    for weights in plan['vote_weights']:
        probability = voted_probabilities(members, validation, weights)
        predictions = np.array(CLASSES)[probability.argmax(axis=1)]
        score = metric_pair(val_labels, predictions)
        candidates.append({'kind': 'vote', 'weights': weights,
                           'macro_f1': score[0], 'accuracy': score[1]})
        if score > best_vote_score:
            best_vote_score, vote_probability, vote_weights = score, probability, weights
        if score > best_score:
            best_score, best_predictions = score, predictions
            best_bundle = {'mode': 'vote', 'members': members, 'weights': weights}
    experts, expert_predictions = [], []
    quad_train, quad_val = fuse(training, 'quad'), fuse(validation, 'quad')
    for pair in [('fungi', 'pest'), ('phytophthora', 'fungi')]:
        mask = np.isin(labels, pair)
        expert = CalibratedClassifierCV(svm_pipeline(
            {'kernel': 'rbf', 'C': 2.0, 'gamma': 'scale', 'class_weight': 'balanced'}),
            cv=group_folds(labels[mask], groups[mask]))
        expert.fit(quad_train[mask], labels[mask])
        experts.append(expert)
        expert_predictions.append(expert.predict(quad_val))
    for first in plan['specialist_thresholds']:
        for second in plan['specialist_thresholds']:
            predictions = apply_specialists(vote_probability, expert_predictions, [first, second])
            score = metric_pair(val_labels, predictions)
            candidates.append({'kind': 'cascade', 'thresholds': [first, second],
                               'macro_f1': score[0], 'accuracy': score[1]})
            if score > best_score:
                best_score, best_predictions = score, predictions
                best_bundle = {'mode': 'cascade', 'members': members, 'weights': vote_weights,
                               'experts': experts, 'thresholds': [first, second]}
    needed = set().union(*(set(REPRESENTATIONS[m['representation']])
                          for m in best_bundle['members']))
    if best_bundle['mode'] == 'cascade':
        needed.update(BACKBONES)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix='.v5-') as temporary:
        staging = Path(temporary) / 'bundle'
        staging.mkdir()
        for name in sorted(needed):
            shutil.copyfile(states[name], staging / f'{name}.pt')
        joblib.dump(best_bundle, staging / 'classifier.joblib', compress=3)
        metadata = {'schema_version': 5, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                    'mode': best_bundle['mode'], 'class_order': list(CLASSES),
                    'seed': 42, 'tta': 'original_and_horizontal_flip_mean',
                    'selected_candidate': next(candidate for candidate in candidates
                                               if (candidate['macro_f1'], candidate['accuracy']) == best_score),
                    'manifest_sha256': manifest_hash, 'training_images': len(train_rows),
                    'validation_images': len(val_rows), 'test_images_used': 0,
                    'plan': plan, 'environment': environment_report(),
                    'backbones': {name: {'weights': str(BACKBONES[name][1]),
                                         'transform': str(BACKBONES[name][1].transforms())}
                                  for name in sorted(needed)},
                    'artifact_sha256': {p.name: sha256(p) for p in staging.iterdir()},
                    'validation_accuracy': best_score[1], 'validation_macro_f1': best_score[0]}
        write_json(staging / 'metadata.json', metadata)
        with tempfile.TemporaryDirectory(dir=report.parent, prefix='.v5-report-') as temporary_report:
            report_stage = Path(temporary_report) / 'report'
            report_stage.mkdir()
            write_validation_reports(report_stage, val_rows, best_predictions, candidates)
            write_json(report_stage / 'run.json', metadata)
            staging.rename(out)
            report_stage.rename(report)
    print(json.dumps({'accuracy': best_score[1], 'macro_f1': best_score[0],
                      'mode': best_bundle['mode'], 'model': str(out), 'report': str(report)}), flush=True)


def predict(model_path, image_path, requested_device='auto'):
    configure_torch()
    root = Path(model_path)
    metadata = json.loads((root / 'metadata.json').read_text(encoding='utf-8'))
    if metadata['schema_version'] != 5 or metadata['class_order'] != list(CLASSES):
        raise ValueError('Unsupported v5 model bundle')
    for filename, expected in metadata['artifact_sha256'].items():
        if Path(filename).name != filename or sha256(root / filename) != expected:
            raise ValueError(f'Invalid model artifact: {filename}')
    device, features = select_device(requested_device), {}
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    for name in metadata['backbones']:
        model = build_extractor(name, root / f'{name}.pt').to(device)
        transform = BACKBONES[name][1].transforms()
        batch = torch.stack([transform(image), transform(ImageOps.mirror(image))]).to(device)
        with torch.inference_mode():
            features[name] = model(batch).mean(0, keepdim=True).cpu().numpy()
        del model
    bundle = joblib.load(root / 'classifier.joblib')
    if bundle['mode'] == 'single':
        member = bundle['members'][0]
        prediction = member['model'].predict(fuse(features, member['representation']))
    else:
        probability = voted_probabilities(bundle['members'], features, bundle['weights'])
        prediction = np.array(CLASSES)[probability.argmax(axis=1)]
        if bundle['mode'] == 'cascade':
            experts = [model.predict(fuse(features, 'quad')) for model in bundle['experts']]
            prediction = apply_specialists(probability, experts, bundle['thresholds'])
    result = {'image': str(image_path), 'predicted_category': str(prediction[0]), 'version': 5}
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description='V5 group-calibrated feature-fusion SVM')
    parser.add_argument('--data', default='data/raw')
    parser.add_argument('--manifest', default='reports/data-v1/manifest.csv')
    parser.add_argument('--cache', default='runs/v5-cache')
    parser.add_argument('--out', default='runs/baseline-v5')
    parser.add_argument('--report', default='reports/baseline-v5')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=2)
    train(parser.parse_args())


if __name__ == '__main__':
    main()
