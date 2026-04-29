import load
import transform
import model
import visualization

if __name__ == '__main__':
    df_raw = load.load_data()
    print("Loaded data")
    df_clean = transform.transform_data(df_raw)
    print("Feature Engineering Complete")
    train, valid, test = model.split_data(df_clean)
    model.baseline_test(train)
    final_model, metrics, importance =model.run_xgboost_pipeline(train, valid, test)
    visualization.run_all_visuals(test,final_model)