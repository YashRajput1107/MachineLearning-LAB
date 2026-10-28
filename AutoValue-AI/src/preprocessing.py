"""Data loading, cleaning, feature engineering and the preprocessing pipeline.

The 7 original input fields are: name, year, km_driven, fuel, seller_type,
transmission, owner. `name` is split into `brand` and `model`, so the model
receives 8 features. The target is `selling_price`.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "Car details v3.csv"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

RANDOM_SEED = 42
TARGET = "selling_price"
INPUT_FIELDS = ["name", "year", "km_driven", "fuel", "seller_type", "transmission", "owner"]
CATEGORICAL = ["brand", "model", "fuel", "seller_type", "transmission", "owner"]
NUMERIC = ["year", "km_driven"]
FEATURES = CATEGORICAL + NUMERIC

MULTI_WORD_BRANDS = ["Land Rover", "Ashok Leyland"]
KM_MAX_PLAUSIBLE = 1_000_000          # readings above 10 lakh km treated as data errors


def load_raw(path=RAW_CSV):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.download_data` first.")
    return pd.read_csv(path)


def split_name(name):
    """'Maruti Swift Dzire VDI' -> ('Maruti', 'Maruti Swift')."""
    name = " ".join(str(name).split())
    for brand in MULTI_WORD_BRANDS:
        if name.startswith(brand + " "):
            rest = name[len(brand) + 1:].split()
            return brand, f"{brand} {rest[0]}" if rest else brand
    parts = name.split()
    brand = parts[0]
    model = f"{parts[0]} {parts[1]}" if len(parts) > 1 else parts[0]
    return brand, model


def add_features(df):
    df = df.copy()
    pairs = df["name"].map(split_name)
    df["brand"] = pairs.str[0]
    df["model"] = pairs.str[1]
    return df


def clean(df):
    """Clean the raw data. Returns (clean_df, report)."""
    report = {"raw_rows": len(df)}
    df = df[INPUT_FIELDS + [TARGET]].copy()

    before = len(df)
    df = df.drop_duplicates()
    report["exact_duplicates_removed"] = before - len(df)

    before = len(df)
    df = df[df["owner"] != "Test Drive Car"]
    report["test_drive_rows_removed"] = before - len(df)

    before = len(df)
    df = df[(df[TARGET] > 0) & (df["km_driven"] >= 0) & (df["km_driven"] <= KM_MAX_PLAUSIBLE)]
    report["invalid_rows_removed"] = before - len(df)

    df = add_features(df).reset_index(drop=True)
    report["clean_rows"] = len(df)
    return df, report


def build_preprocessor(scale_numeric=False):
    """One-hot encode categoricals (unknown categories ignored, rare ones grouped);
    numerics are passed through, or scaled for linear models. Fitted inside the
    model pipeline, i.e. on training data only."""
    numeric = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), CATEGORICAL),
        ("num", numeric, NUMERIC),
    ])


def details_to_frame(details):
    """Turn user-entered details (with `name` or `brand`+`model`) into a 1-row
    feature frame in the column order the pipeline expects."""
    d = dict(details)
    if "brand" not in d or "model" not in d:
        d["brand"], d["model"] = split_name(d["name"])
    return pd.DataFrame([{f: d[f] for f in FEATURES}])


def validate_input(details, meta):
    """Return a list of human-readable problems with the user's input."""
    errors = []
    try:
        year = int(details["year"])
        if not meta["year_min"] <= year <= meta["year_max"]:
            errors.append(f"Manufacturing year must be between {meta['year_min']} and "
                          f"{meta['year_max']} (the range covered by the training data).")
    except (TypeError, ValueError, KeyError):
        errors.append("Manufacturing year must be a whole number.")
    try:
        km = float(details["km_driven"])
        if km < 0:
            errors.append("Kilometres driven cannot be negative.")
        elif km > meta["km_max"]:
            errors.append(f"Kilometres driven above {meta['km_max']:,} is outside the "
                          "training data range.")
    except (TypeError, ValueError, KeyError):
        errors.append("Kilometres driven must be a number.")
    for col in ["fuel", "seller_type", "transmission", "owner", "brand"]:
        if col in details and details[col] not in meta["categories"][col]:
            errors.append(f"Unsupported {col.replace('_', ' ')}: {details[col]!r}.")
    return errors
