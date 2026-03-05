from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any

app = FastAPI(title="Customer Churn Prediction API")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"

model = None
EXPECTED_N_FEATURES = None
FEATURE_NAMES = None


class PredictRequest(BaseModel):
    # Option 1: pass list of features (must match expected length)
    features: Optional[List[float]] = None
    # Option 2: pass dict of feature_name -> value
    data: Optional[Dict[str, Any]] = None


@app.on_event("startup")
def load_model():
    global model, EXPECTED_N_FEATURES, FEATURE_NAMES
    if not MODEL_PATH.exists():
        model = None
        EXPECTED_N_FEATURES = None
        FEATURE_NAMES = None
        return

    model = joblib.load(MODEL_PATH)
    EXPECTED_N_FEATURES = getattr(model, "n_features_in_", None)
    FEATURE_NAMES = getattr(model, "feature_names_in_", None)


@app.get("/")
def root():
    return {"message": "API is running. Go to /docs for Swagger UI."}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "expected_n_features": EXPECTED_N_FEATURES,
        "feature_names_available": FEATURE_NAMES is not None,
        "model_path": str(MODEL_PATH),
    }


def _predict_proba(x_df_or_array):
    # Handles both numpy array and pandas dataframe
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(x_df_or_array)[0, 1])
    # fallback
    return float(model.predict(x_df_or_array)[0])


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not found. Run DVC pipeline to create models/best_model.joblib")

    # ----- Case A: list of features -----
    if req.features is not None:
        if EXPECTED_N_FEATURES is not None and len(req.features) != EXPECTED_N_FEATURES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid number of features. Expected {EXPECTED_N_FEATURES}, got {len(req.features)}",
            )

        x = np.array(req.features, dtype=float).reshape(1, -1)

        # if model has feature names, convert to DataFrame to avoid warnings
        if FEATURE_NAMES is not None:
            x_df = pd.DataFrame(x, columns=list(FEATURE_NAMES))
            prob = _predict_proba(x_df)
        else:
            prob = _predict_proba(x)

        pred = "Yes" if prob >= 0.5 else "No"
        return {"churn_probability": prob, "prediction": pred}

    # ----- Case B: dict of named features -----
    if req.data is not None:
        if FEATURE_NAMES is None:
            raise HTTPException(status_code=400, detail="This model does not expose feature_names_in_. Use 'features' list instead.")

        # build row in correct order
        row = []
        missing = []
        for col in FEATURE_NAMES:
            if col in req.data:
                row.append(req.data[col])
            else:
                missing.append(col)

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing {len(missing)} required features. Example missing: {missing[:5]}",
            )

        x_df = pd.DataFrame([row], columns=list(FEATURE_NAMES))
        prob = _predict_proba(x_df)
        pred = "Yes" if prob >= 0.5 else "No"
        return {"churn_probability": prob, "prediction": pred}

    raise HTTPException(status_code=400, detail="Provide either 'features' (list) or 'data' (dict).")