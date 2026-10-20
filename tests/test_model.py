import numpy as np
import pandas as pd

from src.config import FEATURES
from src.data import clean, split_xy
from src.features import build_preprocessor
from src.predict import predict_records


def test_preprocessor_outputs_no_nans(raw_df):
    X, _ = split_xy(clean(raw_df))
    Xt = build_preprocessor().fit_transform(X)
    assert not np.isnan(Xt.toarray() if hasattr(Xt, "toarray") else Xt).any()
    assert Xt.shape[0] == len(X)


def test_preprocessor_handles_unseen_category(raw_df):
    X, _ = split_xy(clean(raw_df))
    prep = build_preprocessor().fit(X)
    X_new = X.head(2).copy()
    X_new["thal"] = 99  # unseen category must not crash
    assert prep.transform(X_new).shape[0] == 2


def test_probabilities_valid(trained_pipeline, raw_df):
    X, _ = split_xy(clean(raw_df))
    proba = trained_pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1)


def test_predict_records_format(trained_pipeline, raw_df):
    X, _ = split_xy(clean(raw_df))
    out = predict_records(trained_pipeline, X.head(3).to_dict("records"))
    assert len(out) == 3
    for r in out:
        assert r["prediction"] in (0, 1)
        assert 0.5 <= r["confidence"] <= 1.0


def test_model_beats_chance(trained_pipeline, raw_df):
    X, y = split_xy(clean(raw_df))
    assert (trained_pipeline.predict(X) == y).mean() > 0.6
    assert list(pd.DataFrame(X).columns) == FEATURES
