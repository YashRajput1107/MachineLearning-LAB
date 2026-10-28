"""Tests for UCI HAR data loading."""
from src.data_loader import load_train, load_test, load_features
from src.config import ACTIVITY_MAP


def test_train_test_shapes():
    Xtr, ytr, ytr_name, subj = load_train()
    Xte, yte, yte_name, subj_te = load_test()
    assert Xtr.shape == (7352, 561)
    assert Xte.shape == (2947, 561)
    assert list(Xtr.columns) == list(Xte.columns)


def test_feature_names_unique():
    feats = load_features()
    assert len(feats) == 561
    assert len(set(feats)) == 561          # duplicates were de-duplicated


def test_labels_map_to_six_activities():
    _, y, y_name, _ = load_train()
    assert set(y.unique()) == set(ACTIVITY_MAP.keys())
    assert set(y_name.unique()) == set(ACTIVITY_MAP.values())


def test_subjects_disjoint_between_train_and_test():
    _, _, _, subj_tr = load_train()
    _, _, _, subj_te = load_test()
    assert set(subj_tr) & set(subj_te) == set()   # official subject-wise split
