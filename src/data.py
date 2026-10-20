"""Task 1 - data loading and cleaning."""
import logging
from pathlib import Path

import pandas as pd

from src.config import COLUMNS, FEATURES, PROCESSED_PATH, RAW_PATH, TARGET

logger = logging.getLogger(__name__)


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the headerless UCI file; '?' marks missing values."""
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
    logger.info("Loaded raw data: %s", df.shape)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, binarise target (0 = no disease, 1 = disease), drop duplicates.

    Missing values (ca, thal) are deliberately kept as NaN here and imputed inside the
    sklearn Pipeline so that imputation statistics are learnt on training folds only
    (prevents data leakage).
    """
    out = df.copy()
    for col in COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out[TARGET] = (out[TARGET] > 0).astype(int)
    out = out.drop_duplicates().reset_index(drop=True)
    return out


def save_clean(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved cleaned data to %s", path)


def load_clean(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame):
    return df[FEATURES], df[TARGET]


def prepare() -> pd.DataFrame:
    """raw -> clean -> saved CSV."""
    df = clean(load_raw())
    save_clean(df)
    print(f"Clean data: {df.shape}, missing values: {int(df.isna().sum().sum())}")
    print(f"Class balance:\n{df[TARGET].value_counts()}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare()
