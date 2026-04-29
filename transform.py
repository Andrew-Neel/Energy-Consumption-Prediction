import pandas as pd
import numpy as np

def transform_data(df):
    '''
    does feature extraction and renames some columns
    '''
    TARGET = "Value_NE5"
    # -----------------------------------------------------
    # Calendar Features
    # -----------------------------------------------------
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    df["minute_block"] = df.index.minute // 15     # 0,1,2,3
    df["day_of_week"] = df.index.dayofweek         # Mon=0
    df["month"] = df.index.month
    df["weekend"] = (df["day_of_week"] >= 5).astype(int)

    # -----------------------------------------------------
    # Lag Features
    # -----------------------------------------------------
    # 15-min data

    df["demand_15min_ago"] = df[TARGET].shift(1)
    df["demand_1hr_ago"]   = df[TARGET].shift(4)
    df["demand_24hr_ago"]  = df[TARGET].shift(96)
    df["demand_7days_ago"] = df[TARGET].shift(672)

    # -----------------------------------------------------
    # Rolling Features
    # -----------------------------------------------------

    # Last hour = 4 rows
    df["rolling_mean_1hr"] = df[TARGET].rolling(window=4).mean()

    # Last day = 96 rows
    df["rolling_std_1day"] = df[TARGET].rolling(window=96).std()

    # -----------------------------------------------------
    # Weather Features
    # -----------------------------------------------------
    df = df.rename(columns={
        "T [°C]": "temperature",
        "Hr [%Hr]": "humidity",
        "WVs [m/s]": "wind_speed",
        "p [hPa]": "pressure",
        "StrGlo [W/m2]": "solar_radiation",
        "RainDur [min]": "rain_duration",
        "WD [°]": "wind_direction",
        "WVv [m/s]": "wind_speed_vector"

    })

    # Nonlinear temperature
    df["temp_squared"] = df["temperature"] ** 2

    # Feels Like Approximation
    # Simple wind chill / heat stress proxy
    df["feels_like"] = (
        df["temperature"]
        - 0.2 * df["wind_speed"]
        + 0.05 * df["humidity"]
    )

    # -----------------------------------------------------
    # Drop Rows 
    # -----------------------------------------------------
    df = df.dropna()
    df = df.drop(columns='Value_NE7') #drop unused target data to avoid date leaks

    # -----------------------------------------------------
    # FINAL CHECK
    # -----------------------------------------------------
    print(df.head())
    print("\nShape:", df.shape)

    return df
