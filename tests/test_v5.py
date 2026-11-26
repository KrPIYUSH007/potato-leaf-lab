import numpy as np
import pytest

pytest.importorskip('torch')
pytest.importorskip('torchvision')

from leaflab.download import CLASSES
from leaflab.v5 import apply_specialists, cache_settings, group_folds


def test_calibration_keeps_image_groups_together():
    labels = np.tile(CLASSES, 6)
    groups = np.repeat(np.arange(6), len(CLASSES))
    folds = group_folds(labels, groups)
    validation_indices = []
    for train, validation in folds:
        assert not set(groups[train]) & set(groups[validation])
        validation_indices.extend(validation)
    assert sorted(validation_indices) == list(range(len(labels)))


def test_specialists_preserve_first_expert_priority_and_other_categories():
    probability = np.zeros((3, 7))
    probability[0, [1, 4, 5]] = [0.4, 0.35, 0.25]
    probability[1, [1, 4, 5]] = [0.35, 0.25, 0.4]
    probability[2, 0] = 1.0
    experts = [np.array(['pest', 'pest', 'pest']),
               np.array(['phytophthora', 'fungi', 'fungi'])]
    predictions = apply_specialists(probability, experts, [0.1, 0.2])
    assert predictions.tolist() == ['pest', 'fungi', 'bacteria']
    disabled = apply_specialists(probability, experts, [0, 0])
    assert disabled.tolist() == ['fungi', 'phytophthora', 'bacteria']


def test_cache_identity_tracks_manifest_order_and_image_hashes():
    rows = [{'path': 'one.jpg', 'file_sha256': 'aaa'},
            {'path': 'two.jpg', 'file_sha256': 'bbb'}]
    original = cache_settings('convnext', rows, 'manifest')
    assert original != cache_settings('convnext', rows[::-1], 'manifest')
    assert original != cache_settings('convnext', rows, 'changed-manifest')
    changed = [dict(row) for row in rows]
    changed[0]['file_sha256'] = 'different-image'
    assert original != cache_settings('convnext', changed, 'manifest')
