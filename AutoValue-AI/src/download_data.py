"""Download the CarDekho 'Car details v3.csv' dataset into data/raw/.

Original source: Kaggle - "Vehicle dataset from CarDekho" (nehalbirla).
Kaggle requires a login, so the file is fetched from public GitHub mirrors.

Usage:  python -m src.download_data
"""
import urllib.request

from .preprocessing import RAW_CSV

MIRRORS = [
    "https://raw.githubusercontent.com/ayaanqaiserh/car-price-prediction/main/Car%20details%20v3.csv",
    "https://raw.githubusercontent.com/imanishshahu/Car-Price-Prediction/main/Car%20details%20v3.csv",
]


def download(force=False):
    if RAW_CSV.exists() and not force:
        print(f"Dataset already present: {RAW_CSV}")
        return RAW_CSV
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    for url in MIRRORS:
        try:
            print(f"Downloading from {url} ...")
            urllib.request.urlretrieve(url, RAW_CSV)
            print(f"Saved to {RAW_CSV}")
            return RAW_CSV
        except Exception as exc:
            print(f"  failed: {exc}")
    raise RuntimeError("Could not download the dataset from any mirror.")


if __name__ == "__main__":
    download()
