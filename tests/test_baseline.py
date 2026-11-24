"""Regression check that validation data never fits the baseline scaler or SVM."""
import leaflab.prepare
import numpy as np
import pytest

pytest.importorskip('torch')
pytest.importorskip('torchvision')

from leaflab.baseline import fit_candidates
from leaflab.download import CLASSES


def test_baseline_fits_only_training_samples():
    # Shifted validation values make accidental combined-data scaling detectable.
    train_features = np.repeat(np.eye(7) * 10, 4, axis=0)
    train_labels = np.repeat(CLASSES, 4)
    val_features = np.eye(7) * 20
    val_labels = np.array(CLASSES)
    classifier, predictions, candidates = fit_candidates(
        train_features, train_labels, val_features, val_labels)

    scaler = classifier.named_steps['standardscaler']
    np.testing.assert_allclose(scaler.mean_, train_features.mean(axis=0))
    assert int(scaler.n_samples_seen_) == len(train_features)
    svm = classifier.named_steps['svc']
    assert int(svm.n_support_.sum()) <= len(train_features)
    np.testing.assert_array_equal(predictions, val_labels)
    assert len(candidates) == 8
