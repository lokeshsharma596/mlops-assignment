"""Tasks 2-4 - train, tune, evaluate, track with MLflow, and package the best model."""
import argparse
import json
import logging
import os
import platform
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.config import (  # noqa: E402
    FIGURES_DIR, METADATA_PATH, MODEL_DIR, MODEL_PATH, RANDOM_STATE, ROOT, TEST_SIZE,
)
from src.data import load_clean, split_xy  # noqa: E402
from src.features import build_preprocessor  # noqa: E402

logger = logging.getLogger("train")

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

# model name -> (estimator, hyper-parameter grid for the 'clf' step)
CANDIDATES = {
    "logistic_regression": (
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        {"clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l2"],
         "clf__class_weight": [None, "balanced"]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {"clf__n_estimators": [100, 300], "clf__max_depth": [3, 5, None],
         "clf__min_samples_leaf": [1, 3]},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
         "clf__max_depth": [2, 3]},
    ),
}


def evaluate(pipe, X, y) -> dict:
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def save_plots(pipe, X_test, y_test, name: str) -> list:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax)
    ax.set_title(f"ROC - {name}")
    p = FIGURES_DIR / f"roc_{name}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, ax=ax, cmap="Blues")
    ax.set_title(f"Confusion matrix - {name}")
    p = FIGURES_DIR / f"cm_{name}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    clf = pipe.named_steps["clf"]
    importances = getattr(clf, "feature_importances_", None)
    if importances is None and hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    if importances is not None:
        names = pipe.named_steps["prep"].get_feature_names_out()
        order = np.argsort(importances)[-12:]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(np.array(names)[order], importances[order])
        ax.set_title(f"Top features - {name}")
        p = FIGURES_DIR / f"importance_{name}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def main(args) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
    mlflow.set_experiment(args.experiment)

    df = load_clean()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    best = {"name": None, "cv_roc_auc": -1.0, "pipe": None}

    for name, (estimator, grid) in CANDIDATES.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
        with mlflow.start_run(run_name=name) as run:
            gs = GridSearchCV(pipe, grid, scoring=SCORING, refit="roc_auc", cv=cv, n_jobs=-1)
            gs.fit(X_train, y_train)
            idx = gs.best_index_

            mlflow.set_tags({"model_type": name, "dataset": "UCI Heart Disease (Cleveland)"})
            mlflow.log_params({k.replace("clf__", ""): v for k, v in gs.best_params_.items()})
            mlflow.log_params({"cv_folds": args.folds, "test_size": TEST_SIZE,
                               "random_state": RANDOM_STATE, "n_train": len(X_train)})

            cv_metrics = {f"cv_{m}": gs.cv_results_[f"mean_test_{m}"][idx] for m in SCORING}
            cv_std = {f"cv_{m}_std": gs.cv_results_[f"std_test_{m}"][idx] for m in SCORING}
            test_raw = evaluate(gs.best_estimator_, X_test, y_test)
            test_metrics = {f"test_{k}": v for k, v in test_raw.items()}
            mlflow.log_metrics({**cv_metrics, **cv_std, **test_metrics})

            for p in save_plots(gs.best_estimator_, X_test, y_test, name):
                mlflow.log_artifact(str(p), artifact_path="plots")

            signature = infer_signature(X_train, gs.best_estimator_.predict_proba(X_train))
            mlflow.sklearn.log_model(gs.best_estimator_, artifact_path="model",
                                     signature=signature, input_example=X_train.head(3))

            results[name] = {**cv_metrics, **test_metrics, "params": gs.best_params_,
                             "run_id": run.info.run_id}
            logger.info("%s | CV AUC %.3f | test AUC %.3f | test recall %.3f", name,
                        cv_metrics["cv_roc_auc"], test_metrics["test_roc_auc"],
                        test_metrics["test_recall"])

            if cv_metrics["cv_roc_auc"] > best["cv_roc_auc"]:
                best.update(name=name, cv_roc_auc=cv_metrics["cv_roc_auc"], pipe=gs.best_estimator_)

    # ---- Task 4: package the winning model + metadata ----
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipe"], MODEL_PATH)
    metadata = {
        "model_name": best["name"],
        "selection_metric": "cv_roc_auc",
        "metrics": {k: v for k, v in results[best["name"]].items() if k != "params"},
        "params": {k: (v if v is None or isinstance(v, (int, float, str)) else str(v))
                   for k, v in results[best["name"]]["params"].items()},
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=float))
    logger.info("Best model: %s -> %s", best["name"], MODEL_PATH)

    with mlflow.start_run(run_name="summary_best_model"):
        mlflow.set_tag("best_model", best["name"])
        mlflow.log_artifact(str(METADATA_PATH))
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(ROOT / "requirements.txt"))
        mlflow.log_metric("best_cv_roc_auc", best["cv_roc_auc"])
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="heart-disease-classification")
    ap.add_argument("--folds", type=int, default=5)
    main(ap.parse_args())
