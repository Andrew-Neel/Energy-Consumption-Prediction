# Electricity Demand Forecasting Using Machine Learning

## Overview

This project develops a time series forecasting model to predict electricity demand using historical consumption data and weather variables. The goal is to improve upon simple baseline forecasting methods by leveraging machine learning, specifically XGBoost, to capture nonlinear relationships and temporal patterns in energy usage.

The model is trained on high-frequency (15-minute interval) electricity consumption data from Zurich, combined with meteorological features such as temperature, humidity, wind speed, and solar radiation.

---

## Problem Statement

Accurate electricity demand forecasting is critical for energy grid stability, cost efficiency, and resource planning. Traditional baseline approaches (such as using previous time steps) often fail to capture complex dependencies in demand patterns influenced by weather and time-based effects.

This project aims to improve forecasting accuracy using a supervised machine learning approach.

---

## Dataset

- Source: [Kaggle – Zurich Electricity Consumption Dataset](https://www.kaggle.com/datasets/sainideeshk/zurich-electricity-consumption-dataset/data)
- Time Range: 2015–2022  
- Frequency: 15-minute intervals  
- Features include:
  - Electricity demand (multiple zones)
  - Weather variables (temperature, humidity, wind, pressure, solar radiation)
  - Time-based features (hour, day of week, month)

---

## Feature Engineering

The following feature engineering techniques were applied:

- Lag features (15 min, 1 hour, 1 day, 1 week)
- Rolling statistics (mean and standard deviation)
- Calendar features (hour, weekday, month, weekend)
- Weather transformations (e.g., temperature squared, feels-like approximation)

---

## Models

### Baseline Models
- Previous interval value
- Same time yesterday
- Same time last week

### Machine Learning Model
- XGBoost Regressor

Hyperparameters were tuned using grid search over:
- Tree depth
- Learning rate
- Subsampling ratios
- Regularization parameters

---

## Results

| Model                 | RMSE   | MAE    | MAPE (%) |
|----------------------|--------|--------|----------|
| Previous Interval    | 450.23 | 311.796 | 1.22%   |
| Same Time Yesterday  | 4124.43| 2445.878 | 9.31%  |
| Same Time Last Week  | 2027.67| 1188.632 | 4.55%  |
| XGBoost Model        | 232.56 | 170.32 | 0.66%    |

The XGBoost model significantly outperformed all baseline methods, reducing error by approximately 48% compared to the best baseline.

---

## Key Insights

- Recent demand history is the strongest predictor of future demand.
- Weather variables (temperature and solar radiation) improve accuracy.
- Electricity demand shows strong daily and weekly seasonality.
- Tree-based models effectively capture nonlinear temporal patterns.

---

## Visualizations

The project includes:
- Actual vs predicted demand plots
- Feature importance analysis
- Residual error distribution
- Model comparison against baselines
- Daily demand cycle visualization

---

## Tech Stack

- Python
- Pandas / NumPy
- XGBoost
- Scikit-learn
- Matplotlib

