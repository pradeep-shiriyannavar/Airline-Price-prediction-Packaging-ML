import os
import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from airline_prediction.config import config


def duration_to_minutes(duration: str) -> int:
    hours, minutes = 0, 0

    if 'H' in duration:
        hours = int(duration.split('H')[0])
        duration = duration.split('H')[1]

    if 'M' in duration:
        minutes = int(duration.replace('M', ''))

    return hours * 60 + minutes


def preprocess_data(df: pd.DataFrame):
    df = df.copy()

    # Date Features
    df['departureDate'] = pd.to_datetime(df['departureDate'])
    df['departureMonth'] = df['departureDate'].dt.month

    # Time Features
    df['departureHour'] = pd.to_datetime(df['departureTime'], format='%H:%M').dt.hour
    df['arrivalHour'] = pd.to_datetime(df['arrivalTime'], format='%H:%M').dt.hour

    # Duration
    df['flightDurationMins'] = df['flightDuration'].apply(duration_to_minutes)

    # Encoding (Dynamic from config)
    encoders = {}

    for col, encoded_col in config.CATEGORICAL_COLUMNS.items():
        le = LabelEncoder()
        df[encoded_col] = le.fit_transform(df[col])
        encoders[col] = le

    # Drop Columns (from config)
    df_model = df.drop(columns=config.DROP_COLUMNS)

    # Split X & y
    X = df_model.drop(columns=[config.TARGET])
    y = df_model[config.TARGET]

    # Scaling (from config)
    scaler = StandardScaler()
    X[config.TIME_COLUMNS] = scaler.fit_transform(
        X[config.TIME_COLUMNS]
    )

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        shuffle=True
    )

    # Save Artifacts
    os.makedirs(config.ENCODER_PATH, exist_ok=True)
    os.makedirs(config.SCALER_PATH, exist_ok=True)

    for col, encoder in encoders.items():
        joblib.dump(
            encoder,
            config.ENCODER_PATH / f"{col}_encoder.pkl"
        )

    joblib.dump(
        scaler,
        config.SCALER_PATH / "scaler.pkl"
    )

    return X_train, X_test, y_train, y_test