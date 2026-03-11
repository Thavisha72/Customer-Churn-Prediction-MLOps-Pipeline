import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

DAGSHUB_REPO_OWNER = "Thavisha72"
DAGSHUB_REPO_NAME = "Customer-Churn-Prediction-Pipeline"

os.environ["DAGSHUB_ALLOW_OAUTH"] = "false"

dagshub.init(
    repo_owner=DAGSHUB_REPO_OWNER,
    repo_name=DAGSHUB_REPO_NAME,
    mlflow=True,
)

mlflow.set_experiment("customer_churn")


def train_models():

    print("Loading data...")

    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")

    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

   # Trained models to the projects models

    models = {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000),
            {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["lbfgs"]
            }
        ),

        "RandomForest": (
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None]
            }
        ),

        "XGBoost": (
            XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
            ),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5]
            }
        )
    }

    best_f1 = 0
    best_model = None
    best_name = ""

    for name, (model, params) in models.items():

        print(f"\nTraining {name} with Hyperparameter Tuning...")

        with mlflow.start_run(run_name=name):

            grid = GridSearchCV(
                model,
                params,
                cv=3,
                scoring="f1",
                n_jobs=-1
            )

            grid.fit(X_train, y_train)

            best_estimator = grid.best_estimator_

            preds = best_estimator.predict(X_test)
            proba = best_estimator.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            auc = roc_auc_score(y_test, proba)

            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", auc)

            mlflow.sklearn.log_model(
                best_estimator,
                name,
                registered_model_name=name
            )

            print("Best Params:", grid.best_params_)
            print(f"Accuracy: {acc:.3f}")
            print(f"F1 Score: {f1:.3f}")

            if f1 > best_f1:
                best_f1 = f1
                best_model = best_estimator
                best_name = name

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")

    print(f"\nBest model: {best_name} with F1={best_f1:.3f}")
    print("Best model saved to models/best_model.pkl")


if __name__ == "__main__":
    train_models()