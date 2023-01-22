import pytest
from src.dataset.dig_rec_dataset import DigRecDataSet
from pathlib import Path
from src.model import Model
from src.model.layer.random_layer import RandomLayer
from src.model.layer.fully_connected_layer import FullyConnectedLayer
from src.model.layer.activation.relu import Relu
from src.model.layer.activation.soft_max import SoftMax
from src.model.cost_function.squared_cost import SquaredCost

CSV_PATH = Path("test/data/digit-recognizer/train.csv")

SPLIT = 0.5


@pytest.fixture
def dig_rec() -> DigRecDataSet:
    return DigRecDataSet(CSV_PATH, SPLIT)


@pytest.fixture
def model(dig_rec: DigRecDataSet) -> Model:
    X, Y = dig_rec.train.X, dig_rec.train.Y
    model = Model(len(X), SquaredCost)
    model.add_layer(FullyConnectedLayer(20, Relu))
    model.add_layer(FullyConnectedLayer(len(Y), SoftMax))
    return model


def test_shape(dig_rec: DigRecDataSet):
    assert dig_rec.train.X.shape[0] == dig_rec.test.X.shape[0]
    assert dig_rec.train.Y.shape[0] == dig_rec.test.Y.shape[0]
    assert dig_rec.train.X.shape[1] == dig_rec.train.Y.shape[1]
    assert dig_rec.test.X.shape[1] == dig_rec.test.Y.shape[1]
    assert dig_rec.train.X.shape[1] > dig_rec.test.X.shape[1]


def test_predict(dig_rec: DigRecDataSet, model: Model):
    X, Y = dig_rec.test.X, dig_rec.test.Y
    output = model.predict(X)
    assert output.shape == Y.shape
    dig_rec.evaluate(output)


def test_gradient_descent(dig_rec: DigRecDataSet, model: Model):
    X, Y = dig_rec.train.X, dig_rec.train.Y
    model.gradiend_descent(X, Y, 0.1)
