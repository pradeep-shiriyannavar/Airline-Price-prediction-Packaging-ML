import pathlib
# import airline_prediction

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent

DATASET_PATH = "dataset/new_dub_fra_flights.csv"

TARGET = "price"


TEST_SIZE = 0.2
RANDOM_SEED = 42

# Feature Engineering
TIME_COLUMNS = ["departureHour", "arrivalHour"]

DROP_COLUMNS = [
    'source', 'departureAirport', 'arrivalAirport',
    'departureDate', 'departureTime', 'arrivalTime',
    'flightDuration', 'airline', 'currency',
    'travelClass', 'aircraft'
]

CATEGORICAL_COLUMNS = {
    "airline": "airline_encoded",
    "aircraft": "aircraft_encoded",
    "travelClass": "travelClass_encoded"
}

# Model & Artifact Paths
ARTIFACTS_PATH = PACKAGE_ROOT / "artifacts"
ENCODER_PATH = ARTIFACTS_PATH / "encoders"
SCALER_PATH = ARTIFACTS_PATH / "scaler"
MODEL_PATH = ARTIFACTS_PATH / "models"

GBR_N_ESTIMATORS = 300
GBR_LEARNING_RATE = 0.1
GBR_MAX_DEPTH = 5

MODEL_NAME = "gradient_boosting_regressor.pkl"