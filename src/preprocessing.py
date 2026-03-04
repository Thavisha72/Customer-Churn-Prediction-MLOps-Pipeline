from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

INPUT_PATH = RAW_DIR / "ingested.csv"

# Change this if your target column name is different
TARGET_COL = "Churn"

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"[preprocessing] Missing input file: {INPUT_PATH}")

    # Ensure output folders exist
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"[preprocessing] Target column '{TARGET_COL}' not found. Columns: {list(df.columns)}")

    # Drop ID column if present (common in churn datasets)
    for col in ["customerID", "CustomerID", "id", "ID"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Separate X and y
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # Convert y to 0/1 if it is Yes/No
    if y.dtype == "object":
        y = y.map({"No": 0, "Yes": 1}).astype(int)

    # Fix TotalCharges sometimes stored as strings with spaces
    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        X["TotalCharges"] = X["TotalCharges"].fillna(X["TotalCharges"].median())

    # Detect categorical/numerical columns
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Preprocess pipeline
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=False))
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop"
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() == 2 else None
    )

    # Fit preprocessor on train, transform both
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Column names for one-hot encoded output
    feature_names = []
    if len(num_cols) > 0:
        feature_names.extend(num_cols)

    if len(cat_cols) > 0:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
        feature_names.extend(ohe_names)

    # Convert to DataFrame for saving
    X_train_df = pd.DataFrame(X_train_t.toarray() if hasattr(X_train_t, "toarray") else X_train_t, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_t.toarray() if hasattr(X_test_t, "toarray") else X_test_t, columns=feature_names)

    y_train_df = pd.DataFrame({"Churn": y_train.values})
    y_test_df = pd.DataFrame({"Churn": y_test.values})

    # Save outputs (these must exist for DVC)
    X_train_path = PROCESSED_DIR / "X_train.csv"
    X_test_path = PROCESSED_DIR / "X_test.csv"
    y_train_path = PROCESSED_DIR / "y_train.csv"
    y_test_path = PROCESSED_DIR / "y_test.csv"

    X_train_df.to_csv(X_train_path, index=False)
    X_test_df.to_csv(X_test_path, index=False)
    y_train_df.to_csv(y_train_path, index=False)
    y_test_df.to_csv(y_test_path, index=False)

    # Save preprocessor (optional but recommended)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

    print(f"[preprocessing] Saved: {X_train_path}")
    print(f"[preprocessing] Saved: {X_test_path}")
    print(f"[preprocessing] Saved: {y_train_path}")
    print(f"[preprocessing] Saved: {y_test_path}")
    print(f"[preprocessing] Saved preprocessor -> {MODELS_DIR / 'preprocessor.joblib'}")

if __name__ == "__main__":
    main()