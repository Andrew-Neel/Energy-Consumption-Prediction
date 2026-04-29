# ==========================================
# Zurich Electricity Dataset
# Step 2: Load and Inspect Data
# ==========================================

import pandas as pd
import numpy as np

def load_and_preview_data():
    '''
    Loads data and provides some preview stats about the data
    '''
    # -----------------------------
    # Load CSV file
    # -----------------------------
    df = pd.read_csv("zurich_electricity_consumption.csv")

    # -----------------------------
    # Preview dataset
    # -----------------------------
    print("First 5 rows:")
    print(df.head())

    print("\nLast 5 rows:")
    print(df.tail())

    # -----------------------------
    # Basic structure
    # -----------------------------
    print("\nShape of dataset:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    # -----------------------------
    # Parse datetime column
    # -----------------------------
    # Replace 'DateTime' with actual datetime column name if different
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Set datetime as index (recommended for time series)
    df = df.set_index("Timestamp").sort_index()

    print("\nDatetime index preview:")
    print(df.index)

    # -----------------------------
    # Missing values check
    # -----------------------------
    print("\nMissing values by column:")
    print(df.isnull().sum())

    # -----------------------------
    # Duplicate rows check
    # -----------------------------
    print("\nDuplicate rows:", df.duplicated().sum())

    # -----------------------------
    # Check time frequency
    # -----------------------------
    print("\nFirst 10 timestamp differences:")
    print(df.index.to_series().diff().value_counts().head(10))

    # Expected for 15-minute data:
    # 0 days 00:15:00

    # -----------------------------
    # Summary statistics
    # -----------------------------
    print("\nNumerical summary:")
    print(df.describe())

    # -----------------------------
    # Detect gaps in timestamps
    # -----------------------------
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="15min"
    )

    missing_timestamps = full_range.difference(df.index)

    print("\nNumber of missing timestamps:", len(missing_timestamps))

    if len(missing_timestamps) > 0:
        print("\nFirst 10 missing timestamps:")
        print(missing_timestamps[:10])

    # -----------------------------
    # Check for negative values
    # -----------------------------
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        negatives = (df[col] < 0).sum()
        if negatives > 0:
            print(f"{col}: {negatives} negative values")

    # -----------------------------
    # Final overview
    # -----------------------------
    print("\nDate range:")
    print(df.index.min(), "to", df.index.max())

    print("\nRows after processing:", len(df))
    
    return df

def load_data():
    '''
    Just loads the data and updates timestamp column
    '''
    # -----------------------------
    # Load CSV file
    # -----------------------------
    df = pd.read_csv("zurich_electricity_consumption.csv")
    # -----------------------------
    # Parse datetime column
    # -----------------------------
    # Replace 'DateTime' with actual datetime column name if different
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Set datetime as index (recommended for time series)
    df = df.set_index("Timestamp").sort_index()

    print(df.shape)
    return df