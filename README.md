Customer Churn Prediction – End-to-End MLOps Pipeline
📌 Project Overview

This project implements a complete MLOps pipeline for predicting customer churn using the Telco Customer Churn dataset.

The system is fully reproducible, modular, containerized, and production-ready.

It includes:

✅ DVC – Data & pipeline versioning

✅ MLflow – Experiment tracking

✅ Airflow – Workflow orchestration

✅ FastAPI – Model deployment (REST API)

✅ Docker – Containerization

✅ Git – Version control

🏗 System Architecture
Raw Data
   ↓
DVC Pipeline
   ├── data_ingestion
   ├── preprocessing
   ├── training (MLflow logging)
   └── evaluation
          ↓
Best Model Saved
          ↓
FastAPI (Dockerized)
          ↓
REST API Endpoint (/predict)
📂 Project Structure
Customer-Churn-Prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
├── reports/
├── api/
│   └── main.py
│
├── airflow_dags/
│   └── churn_dag.py
│
├── dvc.yaml
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
└── README.md
🧠 Machine Learning Models

The following models are trained and compared:

Logistic Regression

Random Forest

XGBoost / Gradient Boosting

Evaluation Metrics:

Accuracy

Precision

Recall

F1-score

ROC-AUC

The best-performing model is saved as:

models/best_model.joblib
⚙️ Setup Instructions
1️⃣ Clone Repository
git clone <your-repo-url>
cd Customer-Churn-Prediction
2️⃣ Create Virtual Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
🔁 DVC Pipeline
Initialize DVC
dvc init
Run Full Pipeline
dvc repro

This runs:

Data ingestion

Preprocessing

Model training

Evaluation

Generated Outputs

models/best_model.joblib

reports/metrics.json

reports/confusion_matrix.png

reports/roc_curve.png

📊 MLflow Experiment Tracking
Start MLflow UI
mlflow ui

Open in browser:

http://localhost:5000

MLflow logs:

Model parameters

Performance metrics

Model comparisons

Experiment history

🐳 Docker – Run API in Container
Build Docker Image
docker build -t churn-api .
Run Container
docker run -d --name churn_api -p 8000:8000 churn-api
🌐 API Access
Swagger UI
http://localhost:8000/docs
Health Check
GET /health

Response:

{
  "status": "ok",
  "model_loaded": true
}
Predict Endpoint
POST /predict

Request:

{
  "features": [0.5, 1.2, 3.4, ...]
}

Response:

{
  "churn_probability": 0.82,
  "prediction": "Yes"
}
🌊 Airflow Orchestration

The DAG is located in:

airflow_dags/churn_dag.py

To start Airflow (Docker-based):

docker compose -f airflow/docker-compose.airflow.yaml up -d

Access UI:

http://localhost:8080

Login:

Username: admin

Password: admin

The DAG executes the full DVC pipeline automatically.

📦 Docker Compose (Optional – Full Stack)

If using full stack:

docker compose up --build

This runs:

MLflow server

FastAPI service

📊 Reproducibility

The entire project can be reproduced using:

dvc repro
docker build .
docker run ...

All dependencies are pinned in requirements.txt.

🔐 Production-Ready Features

Modular pipeline

Experiment tracking

Containerized deployment

Workflow automation

Versioned datasets

Model artifact management

🎯 Demo Steps (5 Minutes)

Show project structure

Run dvc repro

Show MLflow experiments

Start Docker API

Call /predict endpoint

Trigger Airflow DAG

👩‍💻 Author