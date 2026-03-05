import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib

def preprocess_data():
    print("Starting preprocessing...")
    
    df = pd.read_csv("data/raw/churn_raw.csv")
    
    # Fix TotalCharges column (has spaces, should be numeric)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    # Drop customerID (not useful for prediction)
    df.drop("customerID", axis=1, inplace=True)
    
    print(f"After cleaning: {df.shape[0]} rows")
    
    # Encode all categorical columns
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    print(f"Encoded columns: {categorical_cols}")
    
    # Scale numeric columns
    scaler = StandardScaler()
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # Save scaler for later use in API
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    
    # Split features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    
    print("Preprocessing complete!")
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

if __name__ == "__main__":
    preprocess_data()