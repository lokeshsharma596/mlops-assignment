"""Task 1 - download the UCI Heart Disease (Cleveland) dataset and save the raw CSV.

Usage:  python scripts/download_data.py
Fallback (no internet): manually download `processed.cleveland.data` from
https://archive.ics.uci.edu/dataset/45/heart+disease and save it as
data/raw/heart_disease_raw.csv (no header row, 14 comma-separated columns).
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_URL, RAW_PATH  # noqa: E402


def main() -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, RAW_PATH)
    n_lines = sum(1 for _ in open(RAW_PATH))
    print(f"Saved {n_lines} rows to {RAW_PATH}")


if __name__ == "__main__":
    main()
