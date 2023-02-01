import pytest
from src.model import Model
from src.dataset.trivial_dataset import TrivialDataset
from src.model.cost_function.squared_cost import SquaredCost
from src.model.layer.fully_connected_layer import FullyConnectedLayer
from src.model.layer.activation.soft_max import SoftMax
import logging

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def test_model_on_trivial_data():
    dataset = TrivialDataset(None, None)
    input_size, output_size = dataset.train.shape()
    model = Model(input_size, SquaredCost)
    model.add_layer(FullyConnectedLayer(output_size, SoftMax, SquaredCost, 0.01))
    model.summary()

    for i in range(400):
        model.gradiend_descent(dataset.train.X, dataset.train.Y, 0.1)

    layer = model.layers[0]
    W = layer.W
    b = layer.b
    # x1 increases y1
    assert W[0][0] > 0
    # x2 decreases y1
    assert W[0][1] < 0
    # x3 is unrelated -> regularizer should push towards zero
    assert abs(W[0][2]) < 0.3

    # x1 decreases y2
    assert W[1][0] < 0
    # x2 decreases y2
    assert W[1][1] > 0
    # x3 is unrelated -> regularizer should push towards zero
    assert abs(W[1][2]) < 0.3

    pred = model.run(dataset.test.X)
    acc = dataset.evaluate(pred)
    assert acc > 0.8
