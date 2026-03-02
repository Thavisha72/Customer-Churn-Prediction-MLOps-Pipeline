from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_INPUT = PROJECT_ROOT / "data" / "raw" / "Churn prediction DataSet.csv"
RAW_OUTPUT = PROJECT_ROOT / "data" / "raw" / "ingested.csv"


def main():
    if not RAW_INPUT.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {RAW_INPUT}")

    df = pd.read_csv(RAW_INPUT)

    # Basic validation required for pipeline sanity
    required_cols = {"Churn", "TotalCharges", "MonthlyCharges", "tenure"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_OUTPUT, index=False)
    print(f"[data_ingestion] Saved ingested dataset -> {RAW_OUTPUT} | rows={len(df)} cols={len(df.columns)}")


if __name__ == "__main__":
    main()