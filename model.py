import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from sklearn.model_selection import ParameterGrid
from xgboost import XGBRegressor

TARGET = "Value_NE5" 

def split_data(df):
    '''
    splits data into train text and validate based on predetermined split points
    '''
    print("Dataset Range:")
    print(df.index.min(), "to", df.index.max())

    # -------------------------------------------
    # Split Plan
    # -------------------------------------------
    # Train: 2015-01-01 to 2020-12-31
    # Validation: 2021-01-01 to 2021-12-31
    # Test: 2022-01-01 to 2022-08-31

    train = df.loc["2015-01-01 00:00:00":"2020-12-31 23:45:00"].copy()

    valid = df.loc["2021-01-01 00:00:00":"2021-12-31 23:45:00"].copy()

    test = df.loc["2022-01-01 00:00:00":"2022-08-31 00:00:00"].copy()

    # -------------------------------------------
    # Print Results
    # -------------------------------------------
    print("\nRows in each split:")
    print("Train:", len(train))
    print("Validation:", len(valid))
    print("Test:", len(test))

    print("\nSplit Date Ranges:")
    print("Train:", train.index.min(), "to", train.index.max())
    print("Validation:", valid.index.min(), "to", valid.index.max())
    print("Test:", test.index.min(), "to", test.index.max())

    print("\nFeature Shapes:")
    print("train:", train.shape)
    print("valid:", valid.shape)
    print("test :", test.shape)

    return train, valid, test

def evaluate_forecast(actual, pred):
    '''
    helper function to evalueate baselines
    '''
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    return mae, rmse, mape

def baseline_test(test):
    '''
    add and test 3 baselines for later comparison
    '''
    # -----------------------------------------------------
    # BASELINE 1: Previous Interval Value
    # 15-minute data = previous row
    # -----------------------------------------------------
    test["pred_prev_interval"] = test[TARGET].shift(1)

    # -----------------------------------------------------
    # BASELINE 2: Same Time Yesterday
    # 96 intervals per day (15 min data)
    # -----------------------------------------------------
    test["pred_same_yesterday"] = test[TARGET].shift(96)

    # -----------------------------------------------------
    # BASELINE 3: Same Time Last Week
    # 96 * 7 = 672 intervals
    # -----------------------------------------------------
    test["pred_same_last_week"] = test[TARGET].shift(672)

    # -----------------------------------------------------
    # DROP NaNs created by shifts
    # -----------------------------------------------------
    eval_df = test.dropna(subset=[
        "pred_prev_interval",
        "pred_same_yesterday",
        "pred_same_last_week"
    ]).copy()

    results = []

    models = {
        "Previous Interval": "pred_prev_interval",
        "Same Time Yesterday": "pred_same_yesterday",
        "Same Time Last Week": "pred_same_last_week"
    }

    for model_name, pred_col in models.items():
        mae, rmse, mape = evaluate_forecast(
            eval_df[TARGET],
            eval_df[pred_col]
        )

        results.append({
            "Model": model_name,
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "MAPE (%)": round(mape, 3)
        })

    # -----------------------------------------------------
    # RESULTS TABLE
    # -----------------------------------------------------
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("RMSE")

    print("\nBaseline Model Performance")
    print(results_df)

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------
    results_df.to_csv("baseline_results.csv", index=False)

    print("\nSaved: baseline_results.csv")

def rmse(y_true, y_pred):
    '''
    calculates rmse
    '''
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    '''
    calculates mape
    '''
    y_true = np.where(y_true == 0, 1e-8, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def split_xy(df, target):
    '''
    split data into training and target values
    '''
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    drop_cols = [
        "pred_prev_interval",
        "pred_same_yesterday",
        "pred_same_last_week"
    ]

    X = X.drop(columns=[c for c in drop_cols if c in X.columns])

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.ffill().bfill()

    return X, y

def get_param_grid():
    '''
    return params used for model tuning
    '''
    return {
    "n_estimators": [500, 900],
    "max_depth": [3, 4],
    "learning_rate": [0.03, 0.05],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "min_child_weight": [3, 5],
    "reg_lambda": [1, 3]
}


def tune_xgboost(X_train, y_train, X_valid, y_valid, param_grid=None):
    '''
    perform model tuning and find best parameters
    '''

    if param_grid is None:
        param_grid = get_param_grid()

    grid = list(ParameterGrid(param_grid))

    best_score = np.inf
    best_model = None
    best_params = None
    results = []

    print(f"Running {len(grid)} combinations...\n")

    for i, params in enumerate(grid, start=1):

        print(f"{i}/{len(grid)} -> {params}")

        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            **params
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )

        pred = model.predict(X_valid)

        score_rmse = rmse(y_valid, pred)
        score_mae = mean_absolute_error(y_valid, pred)
        score_mape = mape(y_valid.values, pred)

        results.append({
            **params,
            "RMSE": score_rmse,
            "MAE": score_mae,
            "MAPE": score_mape
        })

        if score_rmse < best_score:
            best_score = score_rmse
            best_model = model
            best_params = params

    results_df = pd.DataFrame(results).sort_values("RMSE")

    return best_model, best_params, results_df


# -----------------------------------------------------
# RETRAIN FINAL MODEL
# -----------------------------------------------------
def retrain_final_model(train, valid, target, best_params):
    '''
    retrain final model based on tuning
    '''

    combined = pd.concat([train, valid]).sort_index()

    X_final, y_final = split_xy(combined, target)

    final_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        **best_params
    )

    final_model.fit(X_final, y_final)

    return final_model


# -----------------------------------------------------
# EVALUATE MODEL
# -----------------------------------------------------
def evaluate_model(model, X_test, y_test):
    '''
    calculate metrics for model evaluation
    '''

    pred = model.predict(X_test)

    results = {
        "RMSE": rmse(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "MAPE": mape(y_test.values, pred)
    }

    return results, pred


# -----------------------------------------------------
# FEATURE IMPORTANCE
# -----------------------------------------------------
def get_feature_importance(model, feature_names):
    '''
    get which values were most important to model 
    '''

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return importance


# -----------------------------------------------------
# SAVE MODEL
# -----------------------------------------------------
def save_model(model, path="xgb_best_model.pkl"):
    joblib.dump(model, path)


# -----------------------------------------------------
# FULL PIPELINE
# -----------------------------------------------------
def run_xgboost_pipeline(train, valid, test, target="Value_NE5"):
    '''
    full model pipeline including data split, model tuning and final model run
    '''

    # Split
    X_train, y_train = split_xy(train, target)
    X_valid, y_valid = split_xy(valid, target)
    X_test, y_test = split_xy(test, target)

    print(X_train.dtypes)
    print(X_train.select_dtypes(exclude=["int64","float64","int32","float32"]).columns)

    # Tune
    best_model, best_params, tuning_results = tune_xgboost(
        X_train, y_train, X_valid, y_valid
    )

    tuning_results.to_csv("xgb_tuning_results.csv", index=False)

    print("\nBest Parameters:")
    print(best_params)

    # Retrain
    final_model = retrain_final_model(
        train, valid, target, best_params
    )

    # Evaluate
    metrics, predictions = evaluate_model(
        final_model, X_test, y_test
    )

    print("\nTest Metrics:")
    print(metrics)

    # Importance
    importance = get_feature_importance(
        final_model, X_test.columns
    )

    importance.to_csv("xgb_feature_importance.csv", index=False)

    # Save model
    save_model(final_model)

    return final_model, metrics, importance