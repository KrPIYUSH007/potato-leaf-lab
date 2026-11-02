"""Audit and split images without separating exact or candidate near-duplicate groups."""
import csv
import hashlib
import json
import platform
from importlib.metadata import version
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageDraw
from sklearn.model_selection import GroupShuffleSplit

from .download import ALIASES, CLASSES, sha256

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
FIELDS = ['path', 'label', 'file_sha256', 'pixel_sha256', 'dhash', 'group', 'split', 'width', 'height']
REPORTED_COUNTS = dict(zip(CLASSES, [569, 748, 201, 68, 611, 347, 532]))


def difference_hash(image):
    values = np.asarray(image.convert('L').resize((9, 8), Image.Resampling.LANCZOS))
    return int(''.join('1' if b else '0' for b in (values[:, 1:] > values[:, :-1]).flat), 2)


def scan(root):
    root = Path(root).resolve()
    records, invalid = [], []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        relative = path.relative_to(root)
        matches = [ALIASES.get(p.lower(), p.lower()) for p in relative.parts[:-1]]
        matches = set(matches) & set(CLASSES)
        if len(matches) != 1:
            raise ValueError(f'Cannot determine exactly one category for {relative}')
        try:
            with Image.open(path) as original:
                image = ImageOps.exif_transpose(original).convert('RGB')
                image.load()
                width, height = image.size
                if min(width, height) < 16:
                    raise ValueError('Image is smaller than 16 pixels')
                pixels = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                dhash = f'{difference_hash(image):016x}'
        except (OSError, ValueError) as exc:
            invalid.append({'path': relative.as_posix(), 'error': str(exc)})
            continue
        records.append({'path': relative.as_posix(), 'label': matches.pop(),
                        'file_sha256': sha256(path), 'pixel_sha256': pixels, 'dhash': dhash,
                        'width': width, 'height': height})
        if len(records) % 500 == 0:
            print(f'Decoded {len(records)} images', flush=True)
    if {r['label'] for r in records} != set(CLASSES):
        raise ValueError('One or more categories have no readable images')
    return records, invalid


def group_images(records, distance=4, manual_groups=None):
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        i, j = find(i), find(j)
        if i != j:
            parent[max(i, j)] = min(i, j)

    hashes = [int(r['dhash'], 16) for r in records]
    candidates = []
    for i, row in enumerate(records):
        for j in range(i):
            exact = row['pixel_sha256'] == records[j]['pixel_sha256']
            distance_ij = (hashes[i] ^ hashes[j]).bit_count()
            if exact or distance_ij <= distance:
                union(i, j)
                candidates.append({'left': records[j]['path'], 'right': row['path'],
                                   'distance': distance_ij, 'exact_pixels': exact,
                                   'labels_match': row['label'] == records[j]['label']})
    if manual_groups:
        path_to_index = {r['path']: i for i, r in enumerate(records)}
        seen = {}
        with Path(manual_groups).open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                if row['path'] not in path_to_index or not row['group']:
                    raise ValueError('Manual groups must name an existing relative path and nonempty group')
                i = path_to_index[row['path']]
                if row['group'] in seen:
                    union(i, seen[row['group']])
                seen[row['group']] = i
    grouped = defaultdict(list)
    for i, record in enumerate(records):
        grouped[find(i)].append(record)
    for members in grouped.values():
        group_id = hashlib.sha256('\n'.join(sorted(r['path'] for r in members)).encode()).hexdigest()[:16]
        for row in members:
            row['group'] = group_id
    return candidates


def unique_images(records):
    seen, keep, excluded = {}, [], []
    for row in records:
        pixel_hash = row['pixel_sha256']
        if pixel_hash in seen:
            first = seen[pixel_hash]
            if row['label'] != first['label']:
                raise ValueError(f'Exact duplicate has conflicting labels: {first["path"]}, {row["path"]}')
            excluded.append({'path': row['path'], 'same_pixels_as': first['path']})
        else:
            seen[pixel_hash] = row
            keep.append(row)
    return keep, excluded


def quarantine_conflicts(records):
    """Exclude every label assignment for identical pixels with conflicting labels."""
    labels = defaultdict(set)
    for row in records:
        labels[row['pixel_sha256']].add(row['label'])
    conflicts = {digest for digest, values in labels.items() if len(values) > 1}
    excluded = [{**row, 'reason': 'identical pixels assigned to multiple categories'}
                for row in records if row['pixel_sha256'] in conflicts]
    retained = [row for row in records if row['pixel_sha256'] not in conflicts]
    return retained, excluded


def assign_splits(records, seed=42):
    """Choose group-disjoint candidates by label balance only, never model performance."""
    y = np.array([CLASSES.index(r['label']) for r in records])
    groups = np.array([r['group'] for r in records])
    n = len(records)
    best, best_score = None, float('inf')
    outer = GroupShuffleSplit(n_splits=100, test_size=.30, random_state=seed)
    for train, held in outer.split(np.zeros(n), y, groups):
        if len(set(groups[held])) < 2:
            continue
        inner = GroupShuffleSplit(n_splits=5, test_size=.50, random_state=seed)
        for val_i, test_i in inner.split(held, y[held], groups[held]):
            ids = (train, held[val_i], held[test_i])
            if any(set(y[index]) != set(range(len(CLASSES))) for index in ids):
                continue
            score = 0.
            for index, fraction in zip(ids, (.70, .15, .15)):
                score += abs(len(index)/n - fraction)
                for category in range(len(CLASSES)):
                    score += abs(np.sum(y[index] == category)/np.sum(y == category) - fraction)
            if score < best_score:
                best, best_score = ids, score
    if best is None:
        raise ValueError('Could not create group-disjoint splits containing every category. Review groups/data.')
    for name, indices in zip(('train', 'val', 'test'), best):
        for i in indices:
            records[int(i)]['split'] = name
    return float(best_score)


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_manifest(path, root=None):
    audit_path = Path(path).with_name('audit.json')
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        if sha256(path) != audit['manifest_sha256']:
            raise ValueError('Manifest changed since the frozen audit; create a deliberate new version')
    with Path(path).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({r['path'] for r in rows}) != len(rows):
        raise ValueError('Manifest is empty or repeats a path')
    groups, pixels, files = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in rows:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or '\\' in row['path'] or ':' in row['path']:
            raise ValueError('Manifest image paths must be safe and relative')
        if row['label'] not in CLASSES or row['split'] not in ('train', 'val', 'test'):
            raise ValueError('Unknown label or split')
        for values, key in ((groups, 'group'), (pixels, 'pixel_sha256'), (files, 'file_sha256')):
            values[row[key]].add(row['split'])
        if root:
            image_path = Path(root) / relative
            if sha256(image_path) != row['file_sha256']:
                raise ValueError(f'Image changed since audit: {relative}')
    if any(len(splits) > 1 for values in (groups, pixels, files) for splits in values.values()):
        raise ValueError('Group or duplicate-image leakage across partitions')
    for split in ('train', 'val', 'test'):
        if {r['label'] for r in rows if r['split'] == split} != set(CLASSES):
            raise ValueError(f'{split} is missing a category')
    return rows


def contacts(root, records, candidates, out):
    """Local inspection images; raw images are never committed to GitHub."""
    canvas = Image.new('RGB', (7*170, 4*190), 'white')
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(CLASSES):
        choices = [r for r in records if r['label'] == label][:4]
        for index, row in enumerate(choices):
            with Image.open(Path(root)/row['path']) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                im.thumbnail((160, 150))
                canvas.paste(im, (column*170, index*190+22))
            draw.text((column*170+3, index*190+3), label, fill='black')
    canvas.save(out/'category_samples.jpg')
    # Bounded review pages; all candidates remain available in the CSV.
    for start in range(0, min(len(candidates), 100), 10):
        page = Image.new('RGB', (800, 10*190), 'white')
        text = ImageDraw.Draw(page)
        for i, pair in enumerate(candidates[start:start+10]):
            for j, side in enumerate(('left', 'right')):
                with Image.open(Path(root)/pair[side]) as im:
                    im = ImageOps.exif_transpose(im).convert('RGB')
                    im.thumbnail((180, 150))
                    page.paste(im, (j*400, i*190+35))
                text.text((j*400, i*190), pair[side][:55], fill='black')
            text.text((200, i*190+150), f"dHash distance {pair['distance']}; exact pixels {pair['exact_pixels']}", fill='black')
        page.save(out/f'near_duplicates_{start//10+1:02}.jpg')


def prepare(root, out, seed=42, distance=4, manual_groups=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError('Output already exists: choose a new directory to preserve frozen manifests')
    if not 0 <= distance <= 64:
        raise ValueError('dHash threshold must be between 0 and 64')
    records, invalid = scan(root)
    counts = dict(Counter(r['label'] for r in records))
    readable_total = len(records)
    records, quarantined = quarantine_conflicts(records)
    candidates = group_images(records, distance, manual_groups)
    unique, excluded = unique_images(records)
    balance = assign_splits(unique, seed)
    out.mkdir(parents=True)
    write_csv(out/'manifest.csv', unique, FIELDS)
    validate_manifest(out/'manifest.csv')
    write_csv(out/'near_duplicate_candidates.csv', candidates,
              ['left', 'right', 'distance', 'exact_pixels', 'labels_match'])
    split_counts = {s: {c: sum(r['split'] == s and r['label'] == c for r in unique)
                        for c in CLASSES} for s in ('train', 'val', 'test')}
    train_counts = split_counts['train']
    total = sum(train_counts.values())
    weights = {c: total/(len(CLASSES)*train_counts[c]) for c in CLASSES}
    report = {
        'dataset_doi': '10.17632/ptz377bwb8.1', 'seed': seed, 'python': platform.python_version(),
        'packages': {name: version(name) for name in ('Pillow', 'numpy', 'scikit-learn')},
        'reported_counts': REPORTED_COUNTS, 'readable_counts': counts,
        'readable_total': readable_total, 'retained_total': len(unique), 'invalid': invalid,
        'quarantined_label_conflicts': quarantined,
        'excluded_exact_duplicates': excluded, 'near_duplicate_candidate_pairs': len(candidates),
        'mixed_label_candidate_pairs': sum(not p['labels_match'] for p in candidates),
        'dhash_threshold': distance, 'group_count': len({r['group'] for r in unique}),
        'split_counts': split_counts, 'balance_score': balance,
        'train_only_class_weights': weights, 'manifest_sha256': sha256(out/'manifest.csv'),
        'manual_groups_supplied': bool(manual_groups),
        'limitations': [
            'dHash is a candidate detector, not a proof of visual duplication or plant identity.',
            'Candidate pairs are conservatively connected and kept in one partition; only exact pixels are deduplicated.',
            'Same-plant/session metadata has not been established. Plant-independent generalisation is not demonstrated.',
            'A held-out split differs from the primary paper\'s stratified 10-fold protocol.',
            'No model has been trained; no classification accuracy is measured in Phase II.']}
    (out/'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (out/'classes.json').write_text(json.dumps(list(CLASSES), indent=2), encoding='utf-8')
    contacts(root, records, candidates, out)
    print(json.dumps({k: report[k] for k in ('readable_total', 'retained_total', 'group_count',
                                            'near_duplicate_candidate_pairs', 'split_counts')}, indent=2))
    return report
