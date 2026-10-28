"""Tests for the feature groups."""
from src.data_loader import load_train, load_features
from src.feature_selection import (accelerometer_features, gyroscope_features,
                                    get_feature_groups)
from src.config import SELECTK


def test_accel_and_gyro_groups():
    feats = load_features()
    acc = accelerometer_features(feats)
    gyro = gyroscope_features(feats)
    assert len(acc) > 0 and len(gyro) > 0
    assert set(acc).issubset(set(feats))
    # gyroscope group must not contain accelerometer-only features
    assert all("Gyro" in f for f in gyro)


def test_feature_groups_sizes():
    Xtr, _, ytr_name, _ = load_train()
    feats = load_features()
    groups = get_feature_groups(Xtr, ytr_name, feats, k=SELECTK)
    assert set(groups) == {"full", "accelerometer", "gyroscope", "reduced"}
    assert len(groups["full"]) == 561
    assert len(groups["reduced"]) == SELECTK
    for cols in groups.values():
        assert set(cols).issubset(set(feats))
