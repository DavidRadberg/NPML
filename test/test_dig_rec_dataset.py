import pytest
from src.dataset.dig_rec_dataset import DigRecDataSet
from pathlib import Path
from src.model import Model
from src.model.layer.random_layer import RandomLayer
from src.model.layer.fully_connected_layer import FullyConnectedLayer
from src.model.layer.activation.relu import Relu
from src.model.layer.activation.soft_max import SoftMax
from src.model.cost_function.squared_cost import SquaredCost
from src.model.layer.optimizer.linear_optimizer import LinearOptimizer

CSV_PATH = Path("test/data/digit-recognizer/train.csv")

SPLIT = 0.5


@pytest.fixture
def dig_rec() -> DigRecDataSet:
    return DigRecDataSet(CSV_PATH, SPLIT)


@pytest.fixture
def model(dig_rec: DigRecDataSet) -> Model:
    input_shape = dig_rec.input_shape()
    output_shape = dig_rec.output_shape()

    model = Model(input_shape, SquaredCost)
    model.add_layer(
        FullyConnectedLayer([20], Relu, SquaredCost, 0.01, LinearOptimizer(0.1))
    )
    model.add_layer(
        FullyConnectedLayer(
            output_shape, SoftMax, SquaredCost, 0.01, LinearOptimizer(0.1)
        )
    )
    return model


def test_shape(dig_rec: DigRecDataSet):
    assert dig_rec.train.X.shape[0] == dig_rec.test.X.shape[0]
    assert dig_rec.train.X.shape[1] == dig_rec.test.X.shape[1]
    assert dig_rec.train.Y.shape[0] == dig_rec.test.Y.shape[0]
    assert dig_rec.train.X.shape[3] == dig_rec.train.Y.shape[1]
    assert dig_rec.test.X.shape[3] == dig_rec.test.Y.shape[1]
    assert dig_rec.train.X.shape[3] > dig_rec.test.X.shape[3]


def test_predict(dig_rec: DigRecDataSet, model: Model):
    X, Y = dig_rec.test.X, dig_rec.test.Y
    output = model.run(X)
    assert output.shape == Y.shape
    dig_rec.evaluate(output)


def test_gradient_descent(dig_rec: DigRecDataSet, model: Model):
    X, Y = dig_rec.train.X, dig_rec.train.Y
    model.gradiend_descent(X, Y)
