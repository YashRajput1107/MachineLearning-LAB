"""Tests for prediction helpers (independent of the saved artifacts)."""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import load_train, load_test
from src.predict import predict_one, validate_feature_csv


def _tiny_model():
    Xtr, _, ytr_name, _ = load_train()
    cols = list(Xtr.columns[:20])
    rf = RandomForestClassifier(n_estimators=30, random_state=0).fit(Xtr[cols], ytr_name)
    return rf, cols


def test_predict_one_returns_valid_label_and_scores():
    rf, cols = _tiny_model()
    Xte, _, yte_name, _ = load_test()
    label, scores = predict_one(rf, Xte[cols].iloc[[0]])
    assert label in set(yte_name.unique())
    assert abs(sum(scores.values()) - 1.0) < 1e-6      # probabilities sum to 1
    assert len(scores) == len(rf.classes_)


def test_csv_validator_rejects_wrong_columns():
    bad = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
    ok, msg, prepared = validate_feature_csv(bad, ["tBodyAcc-mean()-X"])
    assert ok is False and prepared is None


def test_csv_validator_accepts_and_reorders():
    df = pd.DataFrame({"b": [1.0], "a": [2.0], "extra": [9.0]})
    ok, msg, prepared = validate_feature_csv(df, ["a", "b"])
    assert ok is True
    assert list(prepared.columns) == ["a", "b"]        # reordered to expected
