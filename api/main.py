from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Churn Prediction API", version="1.0")

# Load model and scaler
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: float
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running!"}

@app.post("/predict")
def predict(data: CustomerData):
    # Convert to dataframe
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])

    # Scale numeric columns
    df[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.transform(
        df[["tenure", "MonthlyCharges", "TotalCharges"]]
    )

    # Predict
    prob = model.predict_proba(df)[0][1]
    prediction = "Yes" if prob > 0.5 else "No"

    return {
        "churn_probability": round(float(prob), 2),
        "prediction": prediction,
        "message": "This customer is likely to churn!" if prediction == "Yes"
                   else "This customer is likely to stay."
    }

@app.get("/health")
def health():
    return {"status": "healthy"}