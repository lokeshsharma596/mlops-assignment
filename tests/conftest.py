"""Shared fixtures: a synthetic dataset with the same schema as the UCI data (offline tests)."""
import numpy as np
import pandas as pd
import pytest

from src.config import COLUMNS


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "age": rng.integers(30, 78, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(95, 200, n),
        "chol": rng.integers(125, 560, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(70, 200, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.uniform(0, 6, n).round(1),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3, 6, 7], n).astype(float),
    })
    score = (df.age / 80 + df.oldpeak / 4 + df.exang + (df.cp == 4) + df.ca / 3
             - df.thalach / 200 + rng.normal(0, 0.4, n))
    df["target"] = (score > score.median()).astype(int) * rng.integers(1, 3, n)
    df.loc[[3, 17], "ca"] = np.nan
    df.loc[[5], "thal"] = np.nan
    return df[COLUMNS]


@pytest.fixture(scope="session")
def trained_pipeline(raw_df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from src.data import clean, split_xy
    from src.features import build_preprocessor
    X, y = split_xy(clean(raw_df))
    pipe = Pipeline([("prep", build_preprocessor()),
                     ("clf", LogisticRegression(max_iter=1000))])
    return pipe.fit(X, y)
