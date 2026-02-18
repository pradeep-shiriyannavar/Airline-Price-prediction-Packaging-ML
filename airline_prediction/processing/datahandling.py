import os
from airline_prediction.config import config
import pandas as pd

def load_dataset():
    path = config.DATASET_PATH
    _data = pd.read_csv(path)
    return _data