"""Load the official UCI HAR files with proper feature names and labels."""
import pandas as pd

from .config import RAW_DIR, ACTIVITY_MAP


def _make_unique(names):
    """UCI features.txt has duplicate names; make them unique by index suffix."""
    seen, out = {}, []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}__{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


def load_features():
    """561 feature names (de-duplicated, order preserved)."""
    f = pd.read_csv(RAW_DIR / "features.txt", sep=r"\s+", header=None)
    return _make_unique(f[1].tolist())


def load_activity_labels():
    return dict(ACTIVITY_MAP)


def _load_split(split):
    feat = load_features()
    X = pd.read_csv(RAW_DIR / split / f"X_{split}.txt", sep=r"\s+", header=None)
    X.columns = feat
    y = pd.read_csv(RAW_DIR / split / f"y_{split}.txt", header=None)[0]
    subjects = pd.read_csv(RAW_DIR / split / f"subject_{split}.txt", header=None)[0]
    y_name = y.map(ACTIVITY_MAP)
    return X, y, y_name, subjects


def load_train():
    """Returns X (named DataFrame), y (int 1..6), y_name (str), subjects."""
    return _load_split("train")


def load_test():
    return _load_split("test")


if __name__ == "__main__":
    Xtr, ytr, ytr_name, str_ = load_train()
    Xte, yte, yte_name, ste = load_test()
    print("train:", Xtr.shape, "| test:", Xte.shape, "| features:", Xtr.shape[1])
    print("train subjects:", sorted(str_.unique()))
    print("class counts:\n", ytr_name.value_counts())
