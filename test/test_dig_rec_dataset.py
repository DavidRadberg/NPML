import pytest
from src.dataset.dig_rec_dataset import DigRecDataSet
from pathlib import Path
from src.model import Model
from src.model.layer.random_layer import RandomLayer
from src.model.layer.activation.relu import Relu

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


def test_run_model_single(dig_rec: DigRecDataSet):
    X, Y = dig_rec.train.get_example()
    model = Model(len(X))
    model.add_layer(RandomLayer(20, Relu()))
    model.add_layer(RandomLayer(len(Y), Relu()))
    output = model.run(X)
    assert len(output) == len(Y)


def test_run_model(dig_rec: DigRecDataSet):
    X, Y = dig_rec.train.X, dig_rec.train.Y
    model = Model(len(X))
    model.add_layer(RandomLayer(20, Relu()))
    model.add_layer(RandomLayer(len(Y), Relu()))
    output = model.run(X)
    assert output.shape == Y.shape
