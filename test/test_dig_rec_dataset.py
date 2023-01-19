import pytest
from src.dataset.dig_rec_dataset import DigRecDataSet
from pathlib import Path

CSV_PATH = Path("test/data/digit-recognizer/train.csv")

SPLIT = 0.1


@pytest.fixture
def dig_rec() -> DigRecDataSet:
    return DigRecDataSet(CSV_PATH, SPLIT)


def test_shape(dig_rec: DigRecDataSet):
    assert dig_rec.train.X.shape[0] == dig_rec.test.X.shape[0]
    assert dig_rec.train.Y.shape[0] == dig_rec.test.Y.shape[0]
    assert dig_rec.train.X.shape[1] == dig_rec.train.Y.shape[1]
    assert dig_rec.test.X.shape[1] == dig_rec.test.Y.shape[1]
    assert dig_rec.train.X.shape[1] > dig_rec.test.X.shape[1]
