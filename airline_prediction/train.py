import os
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import config
from processing.datahandling import load_dataset
from processing.preprocess import preprocess_data


def evaluation_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def main():

    df = load_dataset()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = GradientBoostingRegressor(
        n_estimators=config.GBR_N_ESTIMATORS,
        learning_rate=config.GBR_LEARNING_RATE,
        max_depth=config.GBR_MAX_DEPTH,
        random_state=config.RANDOM_SEED
    )

    model.fit(X_train, y_train)

    os.makedirs(config.MODEL_PATH, exist_ok=True)

    model_path = config.MODEL_PATH / config.MODEL_NAME
    joblib.dump(model, model_path)

    print(f"Model saved at: {model_path}")

    # Load Model (Deserialization)
    final_model = joblib.load(model_path)

    predictions = final_model.predict(X_test)

    rmse, mae, r2 = evaluation_metrics(y_test, predictions)

    print(f"RMSE is {rmse}")
    print(f"MAE is {mae}")
    print(f"R2 is {r2}")


if __name__ == "__main__":
    main()