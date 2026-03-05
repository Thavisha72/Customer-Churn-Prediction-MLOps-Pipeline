import pandas as pd
import os
import shutil

def ingest_data():
    raw_path = "data/raw/Churn Prediction DataSet.csv"
    output_path = "data/raw/churn_raw.csv"
    
    print("Starting data ingestion...")
    
    # Load dataset
    df = pd.read_csv(raw_path)
    
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"Churn distribution:\n{df['Churn'].value_counts()}")
    
    # Save as working copy
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    
    return df

if __name__ == "__main__":
    ingest_data()