import pytest
from src.dataset_loader.kaggle_dataset_loader import KaggleDatasetLoader

DATA_NAME = "digit-recognizer"
DATA_PATH = "test/data/"
CSV_NAME = "train.csv"


def test_download():
    loader = KaggleDatasetLoader()
    path = loader.download(DATA_NAME, DATA_PATH)
    assert str(path) == f"{DATA_PATH}{DATA_NAME}/{CSV_NAME}"
