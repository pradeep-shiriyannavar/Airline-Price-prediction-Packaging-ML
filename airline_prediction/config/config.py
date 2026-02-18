import pathlib
import os
import airline_prediction

PACKAGE_ROOT = pathlib.PATH(airline_prediction.__file__).resolve().parent

DATASET_PATH = "dataset/new_dub_fra_flights.csv"

TARGET = "price"
TEST_SIZE = 0.2
RANDOM_SEED = 42