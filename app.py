from fastapi import FastAPI
from pydantic import BaseModel
from airline_prediction.predict import predict_price

app = FastAPI()


class FlightInput(BaseModel):
    dep_date: str
    dep_time: str
    arr_time: str
    airline: str
    aircraft: str
    travel_class: str
    stops: int


@app.get("/")
def home():
    return {"message": "Airline Price Prediction API is running"}


@app.post("/predict")
def predict(data: FlightInput):

    price = predict_price(
        data.dep_date,
        data.dep_time,
        data.arr_time,
        data.airline,
        data.aircraft,
        data.travel_class,
        data.stops
    )

    return {"predicted_price": price}