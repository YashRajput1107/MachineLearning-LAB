"""Tests for input validation, deal comparison, formatting and prediction."""
from pathlib import Path

import pytest

from src.predict import compare_deal, format_inr, load_pipeline, predict_price
from src.preprocessing import MODEL_PATH, split_name, validate_input

META = {
    "year_min": 1983, "year_max": 2020, "km_max": 577414,
    "categories": {"fuel": ["Diesel", "Petrol"], "seller_type": ["Individual"],
                   "transmission": ["Manual"], "owner": ["First Owner"], "brand": ["Maruti"]},
}
GOOD = {"brand": "Maruti", "model": "Maruti Swift", "year": 2015, "km_driven": 50000,
        "fuel": "Petrol", "seller_type": "Individual", "transmission": "Manual",
        "owner": "First Owner"}


# --------------------------------------------------------------- validation
def test_valid_input_has_no_errors():
    assert validate_input(GOOD, META) == []


def test_negative_km_rejected():
    errors = validate_input({**GOOD, "km_driven": -10}, META)
    assert any("negative" in e for e in errors)


def test_unsupported_year_rejected():
    errors = validate_input({**GOOD, "year": 2024}, META)
    assert any("between 1983 and 2020" in e for e in errors)


def test_unknown_category_rejected():
    errors = validate_input({**GOOD, "fuel": "Electric"}, META)
    assert any("fuel" in e for e in errors)


def test_split_name_handles_multi_word_brands():
    assert split_name("Maruti Swift Dzire VDI") == ("Maruti", "Maruti Swift")
    assert split_name("Land Rover Discovery Sport") == ("Land Rover", "Land Rover Discovery")


# --------------------------------------------------------------- deal checker
def test_equal_asking_price_is_zero_difference():
    d = compare_deal(500000, 500000)
    assert d["difference"] == 0 and d["difference_pct"] == 0
    assert d["label"] == "Close to estimate"


def test_percentage_formula():
    d = compare_deal(580000, 650000)
    assert d["difference"] == 70000
    assert d["difference_pct"] == pytest.approx(12.07, abs=0.01)
    assert d["label"] == "Potentially above estimate"


def test_asking_below_estimate():
    assert compare_deal(500000, 400000)["label"] == "Potentially below estimate"


def test_within_threshold_is_close():
    assert compare_deal(500000, 520000)["label"] == "Close to estimate"   # +4%


def test_compare_rejects_nonpositive_prediction():
    with pytest.raises(ValueError):
        compare_deal(0, 100000)


# --------------------------------------------------------------- formatting
def test_format_inr_indian_grouping():
    assert format_inr(580000) == "₹5,80,000"
    assert format_inr(10000000) == "₹1,00,00,000"
    assert format_inr(999) == "₹999"


# --------------------------------------------------------------- model
def test_missing_model_file_gives_helpful_error():
    with pytest.raises(FileNotFoundError, match="src.train"):
        load_pipeline(Path("does_not_exist.joblib"))


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained yet")
def test_prediction_is_positive_and_plausible():
    price = predict_price(GOOD)
    assert 50_000 < price < 2_000_000


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained yet")
def test_newer_car_differs_from_older_car():
    old = predict_price({**GOOD, "year": 2008})
    new = predict_price({**GOOD, "year": 2018})
    assert old != new
