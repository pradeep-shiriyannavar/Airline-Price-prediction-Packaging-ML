import joblib
import pandas as pd
from datetime import datetime
from airline_prediction.config import config

# Load artifacts once
model = joblib.load(config.MODEL_PATH / config.MODEL_NAME)
scaler = joblib.load(config.SCALER_PATH / "scaler.pkl")

encoders = {
    "airline": joblib.load(config.ENCODER_PATH / "airline_encoder.pkl"),
    "aircraft": joblib.load(config.ENCODER_PATH / "aircraft_encoder.pkl"),
    "travelClass": joblib.load(config.ENCODER_PATH / "travelClass_encoder.pkl"),
}


def calculate_duration(dep_time, arr_time):
    dep = datetime.strptime(dep_time, "%H:%M")
    arr = datetime.strptime(arr_time, "%H:%M")

    duration = (arr - dep).total_seconds() / 60

    # Handle overnight flights
    if duration < 0:
        duration += 24 * 60

    return int(duration)


def predict_price(dep_date, dep_time, arr_time,
                  airline, aircraft, travel_class, stops):


    departure_date = pd.to_datetime(dep_date)

    departure_month = departure_date.month
    departure_hour = datetime.strptime(dep_time, "%H:%M").hour
    arrival_hour = datetime.strptime(arr_time, "%H:%M").hour
    flight_duration_mins = calculate_duration(dep_time, arr_time)


    data = {
        "airline": airline,
        "aircraft": aircraft,
        "travelClass": travel_class,
        "departureMonth": departure_month,
        "departureHour": departure_hour,
        "arrivalHour": arrival_hour,
        "flightDurationMins": flight_duration_mins,
        "numberOfStops": stops
    }

    df = pd.DataFrame([data])

    for col, encoder in encoders.items():
        df[f"{col}_encoded"] = encoder.transform(df[col])


    df = df.drop(columns=["airline", "aircraft", "travelClass"])


    df[config.TIME_COLUMNS] = scaler.transform(df[config.TIME_COLUMNS])

    prediction = model.predict(df)

    return round(float(prediction[0]), 2)

if __name__ == "__main__":
    dep_date = "2025-04-15"
    dep_time = "09:35"
    arr_time = "19:00"
    airline = "IB"
    aircraft = "321"
    travel_class = "ECONOMY"
    stops = 1

    predicted_price = predict_price(
        dep_date, dep_time, arr_time,
        airline, aircraft, travel_class, stops
    )

    print(f"Predicted Price: €{predicted_price}")