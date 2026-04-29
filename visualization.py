import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------
# Helper: prepare test features and predictions
# -----------------------------------------------------
def prepare_predictions(test, model, target="Value_NE5"):
    X_test = test.drop(columns=[target]).copy()
    y_test = test[target].copy()

    drop_cols = [
        "pred_prev_interval",
        "pred_same_yesterday",
        "pred_same_last_week"
    ]

    X_test = X_test.drop(
        columns=[c for c in drop_cols if c in X_test.columns],
        errors="ignore"
    )


    pred = model.predict(X_test)

    results = pd.DataFrame({
        "Actual": y_test,
        "Predicted": pred
    }, index=test.index)

    return X_test, results


# =====================================================
# Actual vs Predicted (Date Range)
# =====================================================
def plot_actual_vs_predicted(
    test,
    model,
    target="Value_NE5",
    start_date="2022-03-01",
    end_date="2022-03-07"
):
    _, results = prepare_predictions(test, model, target)

    plot_df = results.loc[start_date:end_date]

    plt.figure(figsize=(14, 6))
    plt.plot(plot_df.index, plot_df["Actual"], label="Actual")
    plt.plot(plot_df.index, plot_df["Predicted"], label="Predicted")
    plt.title("Actual vs Predicted Electricity Demand")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =====================================================
# RMSE Comparison Chart
# =====================================================
def plot_model_comparison():
    models = [
        "Previous Interval",
        "Same Last Week",
        "Same Yesterday",
        "XGBoost"
    ]

    rmse_vals = [
        450.23,
        2027.67,
        4124.43,
        232.56
    ]

    plt.figure(figsize=(10, 6))
    plt.bar(models, rmse_vals)
    plt.title("RMSE Comparison: Baselines vs XGBoost")
    plt.ylabel("RMSE")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()


# =====================================================
# Feature Importance
# =====================================================
def plot_feature_importance(test, model, target="Value_NE5", top_n=10):
    X_test, _ = prepare_predictions(test, model, target)

    importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": model.feature_importances_
    })

    importance = (
        importance
        .sort_values("Importance", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 6))
    plt.barh(
        importance["Feature"],
        importance["Importance"]
    )
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


# =====================================================
# Residual Histogram
# =====================================================
def plot_residual_distribution(test, model, target="Value_NE5"):
    _, results = prepare_predictions(test, model, target)

    residuals = results["Actual"] - results["Predicted"]

    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50)
    plt.title("Prediction Error Distribution")
    plt.xlabel("Actual - Predicted")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# =====================================================
# Average Demand by Hour
# =====================================================
def plot_avg_hourly_demand(test, target="Value_NE5"):
    df = test.copy()

    df["hour"] = df.index.hour

    hourly = df.groupby("hour")[target].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(hourly.index, hourly.values, marker="o")
    plt.title("Average Electricity Demand by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Average Demand")
    plt.xticks(range(24))
    plt.tight_layout()
    plt.show()


# =====================================================
# Run All Visuals
# =====================================================
def run_all_visuals(test, model, target="Value_NE5"):
    plot_actual_vs_predicted(test, model, target)
    plot_model_comparison()
    plot_feature_importance(test, model, target)
    plot_residual_distribution(test, model, target)
    plot_avg_hourly_demand(test, target)