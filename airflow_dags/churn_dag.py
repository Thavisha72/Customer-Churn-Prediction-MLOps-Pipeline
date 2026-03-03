from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# Project paths (works inside container if you mount project to /opt/project)
PROJECT_ROOT = Path("/opt/project")

RAW_INPUT = PROJECT_ROOT / "data" / "raw" / "telco_customer_churn_data.csv"
INGESTED = PROJECT_ROOT / "data" / "raw" / "ingested.csv"

X_TRAIN = PROJECT_ROOT / "data" / "processed" / "X_train.csv"
X_TEST = PROJECT_ROOT / "data" / "processed" / "X_test.csv"
Y_TRAIN = PROJECT_ROOT / "data" / "processed" / "y_train.csv"
Y_TEST = PROJECT_ROOT / "data" / "processed" / "y_test.csv"

BEST_MODEL = PROJECT_ROOT / "models" / "best_model.joblib"
METRICS = PROJECT_ROOT / "reports" / "metrics.json"
REGISTRY = PROJECT_ROOT / "models" / "registry.json"


def _run_python(script_rel_path: str) -> None:
    """Run a project python script (like src/train.py) in the same container."""
    import subprocess
    script_path = PROJECT_ROOT / script_rel_path
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    # Use python available in the Airflow container
    result = subprocess.run(
        ["python", str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed running {script_rel_path}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


# 1) Data ingestion
def data_ingestion() -> None:
    _run_python("src/data_ingestion.py")
    if not INGESTED.exists():
        raise FileNotFoundError(f"Ingestion output not found: {INGESTED}")


# 2) Data validation
def data_validation() -> None:
    import pandas as pd

    if not INGESTED.exists():
        raise FileNotFoundError(f"Missing ingested data: {INGESTED}")

    df = pd.read_csv(INGESTED)

    required_cols = {"Churn", "TotalCharges", "MonthlyCharges", "tenure"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Dataset is empty!")

    # Validate target values
    if not set(df["Churn"].dropna().unique()).issubset({"Yes", "No"}):
        raise ValueError("Churn column must contain only Yes/No values.")

    # Example: basic null check (allow some nulls but not all)
    null_ratio = df.isna().mean().max()
    if null_ratio > 0.95:
        raise ValueError("Some column has >95% missing values (bad data).")


# 3) Feature engineering (your preprocessing)
def feature_engineering() -> None:
    _run_python("src/preprocessing.py")
    for p in [X_TRAIN, X_TEST, Y_TRAIN, Y_TEST]:
        if not p.exists():
            raise FileNotFoundError(f"Missing preprocessing output: {p}")


# 4) Model training
def model_training() -> None:
    _run_python("src/train.py")
    if not BEST_MODEL.exists():
        raise FileNotFoundError(f"Missing trained model: {BEST_MODEL}")


# 5) Model evaluation
def model_evaluation() -> None:
    _run_python("src/evaluate.py")
    if not METRICS.exists():
        raise FileNotFoundError(f"Missing evaluation metrics: {METRICS}")


# 6) Model registration (simple local registry file)
def model_registration() -> None:
    import time

    if not BEST_MODEL.exists():
        raise FileNotFoundError(f"Model not found for registration: {BEST_MODEL}")
    if not METRICS.exists():
        raise FileNotFoundError(f"Metrics not found for registration: {METRICS}")

    # Read metrics
    with open(METRICS, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Create a simple registry entry (for assignment)
    entry = {
        "model_path": str(BEST_MODEL),
        "registered_at_utc": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "version": int(time.time()),  # simple versioning
        "stage": "Production",
    }

    REGISTRY.parent.mkdir(parents=True, exist_ok=True)

    # Append to registry.json
    if REGISTRY.exists():
        with open(REGISTRY, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if not isinstance(existing, list):
            existing = [existing]
    else:
        existing = []

    existing.append(entry)

    with open(REGISTRY, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


with DAG(
    dag_id="customer_churn_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["churn", "mlops"],
) as dag:

    t1_ingest = PythonOperator(
        task_id="data_ingestion",
        python_callable=data_ingestion,
    )

    t2_validate = PythonOperator(
        task_id="data_validation",
        python_callable=data_validation,
    )

    t3_features = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering,
    )

    t4_train = PythonOperator(
        task_id="model_training",
        python_callable=model_training,
    )

    t5_eval = PythonOperator(
        task_id="model_evaluation",
        python_callable=model_evaluation,
    )

    t6_register = PythonOperator(
        task_id="model_registration",
        python_callable=model_registration,
    )

    # Proper dependencies:
    t1_ingest >> t2_validate >> t3_features >> t4_train >> t5_eval >> t6_register