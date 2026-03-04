import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


# -------------------------
# MLflow setup (local folder to avoid /mlflow permission error)
# -------------------------
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
mlflow.set_experiment("customer_churn")


def load_preprocessed():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")

    y_train_df = pd.read_csv(PROCESSED_DIR / "y_train.csv")
    y_test_df = pd.read_csv(PROCESSED_DIR / "y_test.csv")

    # y should be a Series
    y_train = y_train_df["Churn"] if "Churn" in y_train_df.columns else y_train_df.iloc[:, 0]
    y_test = y_test_df["Churn"] if "Churn" in y_test_df.columns else y_test_df.iloc[:, 0]

    # If labels are strings, convert to 0/1
    if y_train.dtype == "object":
        y_train = y_train.map({"No": 0, "Yes": 1}).astype(int)
    if y_test.dtype == "object":
        y_test = y_test.map({"No": 0, "Yes": 1}).astype(int)

    return X_train, y_train, X_test, y_test


def compute_metrics(model, X, y):
    y_pred = model.predict(X)

    roc = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
        roc = float(roc_auc_score(y, y_proba))

    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": roc,
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_preprocessed()

    candidates = [
        (
            "logistic_regression",
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"max_iter": 2000, "class_weight": "balanced"},
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample",
            ),
            {"n_estimators": 300, "random_state": 42, "class_weight": "balanced_subsample"},
        ),
        (
            "gradient_boosting",
            GradientBoostingClassifier(random_state=42),
            {"random_state": 42},
        ),
    ]

    best_name = None
    best_model = None
    best_score = -1.0
    results = []

    for name, model, params in candidates:
        with mlflow.start_run(run_name=name):
            mlflow.log_params(params)

            model.fit(X_train, y_train)

            metrics = compute_metrics(model, X_test, y_test)
            # log only non-None metrics
            mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})

            # ✅ log model to MLflow (new API style)
            mlflow.sklearn.log_model(model, name="model")

            results.append({"model": name, **metrics})

            # choose best by roc_auc if available, else f1
            score = metrics["roc_auc"] if metrics["roc_auc"] is not None else metrics["f1"]
            if score > best_score:
                best_score = score
                best_name = name
                best_model = model

    # ✅ Save best model for DVC output
    best_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, best_path)

    # Save report
    report_path = REPORTS_DIR / "model_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {"best_model": best_name, "best_score": best_score, "results": results},
            f,
            indent=2,
        )

    print(f"[training] Best model: {best_name} | score={best_score:.4f}")
    print(f"[training] Saved -> {best_path}")
    print(f"[training] Saved -> {report_path}")
    print(f"[mlflow] Tracking dir -> {MLRUNS_DIR}")


if __name__ == "__main__":
    main()