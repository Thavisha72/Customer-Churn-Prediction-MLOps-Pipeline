from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(title="Customer Churn Prediction API")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"

model = None


class PredictRequest(BaseModel):
    features: list[float]


@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        model = None


@app.get("/")
def root():
    return {"message": "API is running. Go to /docs for Swagger UI."}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        return {"error": "Model not found. Run dvc repro to create models/best_model.joblib"}

    x = np.array(req.features, dtype=float).reshape(1, -1)
    prob = float(model.predict_proba(x)[0, 1]) if hasattr(model, "predict_proba") else float(model.predict(x)[0])
    pred = "Yes" if prob >= 0.5 else "No"
    return {"churn_probability": prob, "prediction": pred}