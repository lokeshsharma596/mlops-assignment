"""Task 4 - inference helper (also used as a CLI)."""
import json
import sys

import joblib
import pandas as pd

from src.config import FEATURES, MODEL_PATH


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict_records(model, records: list) -> list:
    """records: list of dicts with the 13 feature keys. Returns prediction + confidence."""
    X = pd.DataFrame(records)[FEATURES]
    proba = model.predict_proba(X)[:, 1]
    return [
        {"prediction": int(p >= 0.5),
         "confidence": float(p if p >= 0.5 else 1 - p),
         "probability_disease": float(p)}
        for p in proba
    ]


if __name__ == "__main__":
    sample = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
        "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}
    print(predict_records(load_model(), [sample]))
