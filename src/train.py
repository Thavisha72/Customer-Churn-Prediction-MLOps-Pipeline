import os
import json
from pathlib import Path
import tempfile

PROJECT_ROOT = os.path.abspath(".")
MLRUNS_DIR = os.path.join(PROJECT_ROOT, "mlruns")

# Use SQLite for tracking
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{MLRUNS_DIR}/mlruns.db"

# Use local folder for all artifacts
os.environ["MLFLOW_ARTIFACT_URI"] = f"file://{MLRUNS_DIR}"

# Force temp files into a writable folder
tempfile.tempdir = os.path.join(MLRUNS_DIR, "tmp")
os.makedirs(tempfile.tempdir, exist_ok=True)

# -------------------------
# Imports
# -------------------------
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
ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
MLRUNS_DIR = ROOT_DIR / "mlruns"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# MLflow setup
# -------------------------
mlflow.set_tracking_uri(f"sqlite:///{MLRUNS_DIR}/mlruns.db")
mlflow.set_experiment("customer_churn")

# -------------------------
# Load preprocessed data
# -------------------------
def load_preprocessed():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")

    y_train_df = pd.read_csv(PROCESSED_DIR / "y_train.csv")
    y_test_df = pd.read_csv(PROCESSED_DIR / "y_test.csv")

    y_train = y_train_df["Churn"] if "Churn" in y_train_df.columns else y_train_df.iloc[:, 0]
    y_test = y_test_df["Churn"] if "Churn" in y_test_df.columns else y_test_df.iloc[:, 0]

    # Convert labels if needed
    if y_train.dtype == "object":
        y_train = y_train.map({"No": 0, "Yes": 1}).astype(int)
    if y_test.dtype == "object":
        y_test = y_test.map({"No": 0, "Yes": 1}).astype(int)

    return X_train, y_train, X_test, y_test

# -------------------------
# Metrics computation
# -------------------------
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

# -------------------------
# Training loop
# -------------------------
def main():
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
            {"n_estimators": 300, "random_state": 42},
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
            # Log parameters
            mlflow.log_params(params)

            # Train model
            model.fit(X_train, y_train)

            # Compute and log metrics
            metrics = compute_metrics(model, X_test, y_test)
            mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})

            # -------------------------
            # Safe MLflow model logging (skops)
            # -------------------------
            mlflow.sklearn.log_model(
                model,
                artifact_path="model"
            )

            results.append({"model": name, **metrics})

            # Track best model
            score = metrics["roc_auc"] if metrics["roc_auc"] is not None else metrics["f1"]
            if score > best_score:
                best_score = score
                best_name = name
                best_model = model

    # -------------------------
    # Save best model locally
    # -------------------------
    best_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, best_path)

    report_path = REPORTS_DIR / "model_results.json"
    with open(report_path, "w") as f:
        json.dump(
            {
                "best_model": best_name,
                "best_score": best_score,
                "results": results
            },
            f,
            indent=2,
        )

    print(f"Best model: {best_name} | score={best_score:.4f}")
    print(f"Saved model -> {best_path}")
    print(f"Saved report -> {report_path}")

if __name__ == "__main__":
    main()