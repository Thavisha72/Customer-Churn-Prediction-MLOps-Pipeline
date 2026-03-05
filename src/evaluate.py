import pandas as pd
import joblib
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score, classification_report)

def evaluate_model():
    print("Evaluating best model...")

    model = joblib.load("models/best_model.pkl")

    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print("\n===== Final Model Evaluation =====")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.3f}")
    print(f"F1 Score:  {f1_score(y_test, preds):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, proba):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds,
                                target_names=["No Churn", "Churn"]))

if __name__ == "__main__":
    evaluate_model()