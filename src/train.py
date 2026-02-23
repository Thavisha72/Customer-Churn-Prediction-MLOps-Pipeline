import os
import json
from pathlib import pathlib

import joblib 
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradiantBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT/ "data/processed"
MODEL_DIR = PROJECT_ROOT/ "models"
REPORTS_DIR = PROJECT_ROOT/ "reports"

def load_prepocessed():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")

    if y_train.dtype == "object":
        y_train = y_train.map({"No": 0, "Yes": 1}).astype(int)
    if y_test.dtype == "object":
        y_test = y_test.map({"No": 0, "Yes": 1}).astype(int)
   
    return X_train, y_train, X_test, y_test


def binary_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    





































