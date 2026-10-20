import pandas as pd

from src.config import COLUMNS, FEATURES, TARGET
from src.data import clean, load_raw, split_xy


def test_clean_binarises_target(raw_df):
    out = clean(raw_df)
    assert set(out[TARGET].unique()) <= {0, 1}


def test_clean_keeps_schema_and_no_duplicates(raw_df):
    out = clean(raw_df)
    assert list(out.columns) == COLUMNS
    assert not out.duplicated().any()


def test_missing_values_are_preserved_for_pipeline(raw_df):
    out = clean(raw_df)
    assert out[["ca", "thal"]].isna().sum().sum() > 0


def test_load_raw_parses_question_marks(tmp_path):
    f = tmp_path / "raw.csv"
    f.write_text("63,1,1,145,233,1,2,150,0,2.3,3,?,6,0\n")
    df = load_raw(f)
    assert df.shape == (1, 14)
    assert pd.isna(df.loc[0, "ca"])


def test_split_xy(raw_df):
    X, y = split_xy(clean(raw_df))
    assert list(X.columns) == FEATURES
    assert len(X) == len(y)
