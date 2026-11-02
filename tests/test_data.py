import hashlib
import random

import pytest
from PIL import Image

from leaflab.download import CLASSES, download_one
from leaflab.prepare import (FIELDS, assign_splits, group_images, prepare,
                             unique_images, validate_manifest, write_csv, quarantine_conflicts)


def record(label, index):
    key = f'{label}-{index}'
    digest = hashlib.sha256(key.encode()).hexdigest()
    return {'path': f'{label}/{index}.png', 'label': label, 'file_sha256': digest,
            'pixel_sha256': digest, 'dhash': digest[:16], 'group': key,
            'width': 32, 'height': 32}


def test_same_pixels_deduplicated_but_conflicting_labels_fail():
    a, b = record('bacteria', 1), record('bacteria', 2)
    b['pixel_sha256'] = a['pixel_sha256']
    keep, excluded = unique_images([a, b])
    assert len(keep) == 1 and len(excluded) == 1
    b['label'] = 'virus'
    with pytest.raises(ValueError, match='conflicting labels'):
        unique_images([a, b])


def test_near_duplicate_connections_are_transitive():
    rows = [record('fungi', i) for i in range(3)]
    for row, value in zip(rows, (0, 1, 3)):
        row['dhash'] = f'{value:016x}'
    candidates = group_images(rows, distance=1)
    assert len(candidates) == 2
    assert len({r['group'] for r in rows}) == 1


def test_quarantine_removes_all_conflicting_copies():
    a, b, c = record('fungi', 0), record('pest', 0), record('healthy', 1)
    b['pixel_sha256'] = a['pixel_sha256']
    keep, excluded = quarantine_conflicts([a, b, c])
    assert keep == [c]
    assert {r['label'] for r in excluded} == {'fungi', 'pest'}


def test_grouped_split_is_reproducible_and_disjoint(tmp_path):
    rows = [record(c, i) for c in CLASSES for i in range(24)]
    for c in CLASSES:
        selected = [r for r in rows if r['label'] == c]
        selected[0]['group'] = selected[1]['group']
    assign_splits(rows, seed=42)
    path = tmp_path/'manifest.csv'
    write_csv(path, rows, FIELDS)
    assert len(validate_manifest(path)) == len(rows)
    first = [r['split'] for r in rows]
    assign_splits(rows, seed=42)
    assert first == [r['split'] for r in rows]
    assignments = {r['path']: r['split'] for r in rows}
    for c in CLASSES:
        assert assignments[f'{c}/0.png'] == assignments[f'{c}/1.png']


def test_leakage_and_unsafe_paths_rejected(tmp_path):
    rows = [record(c, i) for c in CLASSES for i in range(12)]
    assign_splits(rows)
    path = tmp_path/'manifest.csv'
    train = next(r for r in rows if r['split'] == 'train')
    test = next(r for r in rows if r['split'] == 'test')
    test['group'] = train['group']
    write_csv(path, rows, FIELDS)
    with pytest.raises(ValueError, match='leakage'):
        validate_manifest(path)
    test['group'] = 'different'
    test['path'] = '../outside.png'
    write_csv(path, rows, FIELDS)
    with pytest.raises(ValueError, match='safe and relative'):
        validate_manifest(path)


def test_prepare_real_files_detects_corruption_and_duplicates(tmp_path):
    root = tmp_path/'raw'
    rng = random.Random(42)
    for label in CLASSES:
        folder = root/label
        folder.mkdir(parents=True)
        for index in range(12):
            image = Image.frombytes('RGB', (32, 32), rng.randbytes(32*32*3))
            image.save(folder/f'{index}.png')
    (root/'fungi'/'duplicate.png').write_bytes((root/'fungi'/'0.png').read_bytes())
    (root/'pest'/'conflicting.png').write_bytes((root/'fungi'/'1.png').read_bytes())
    (root/'fungi'/'broken.jpg').write_bytes(b'broken image')
    out = tmp_path/'report'
    report = prepare(root, out, distance=0)
    assert report['retained_total'] == 83
    assert len(report['quarantined_label_conflicts']) == 2
    assert len(report['invalid']) == 1
    assert len(report['excluded_exact_duplicates']) == 1
    assert len(validate_manifest(out/'manifest.csv', root)) == 83
    with pytest.raises(FileExistsError):
        prepare(root, out)
    first = next(root.rglob('0.png'))
    first.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed since audit'):
        validate_manifest(out/'manifest.csv', root)
    with (out/'manifest.csv').open('a', encoding='utf-8') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='Manifest changed'):
        validate_manifest(out/'manifest.csv')


def test_downloader_reuses_only_verified_files(tmp_path):
    path = tmp_path/'healthy'/'image.jpg'
    path.parent.mkdir()
    path.write_bytes(b'example')
    item = {'path': 'healthy/image.jpg', 'bytes': 7,
            'sha256': hashlib.sha256(b'example').hexdigest(), 'url': 'unused'}
    assert download_one(item, tmp_path) == 'verified existing'
