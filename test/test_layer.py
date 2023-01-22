from src.model.layer.random_layer import RandomLayer
from src.model.layer.activation.relu import Relu
from src.model.layer.fully_connected_layer import FullyConnectedLayer
from src.model.cost_function.squared_cost import SquaredCost
import numpy as np


def test_shape():
    layer = RandomLayer(10, Relu, SquaredCost, 0.01)
    layer.random_init(10)
    assert layer.input_size == 10
    assert layer.output_size == 10


def test_fully_connected_layer():
    layer = FullyConnectedLayer(2, Relu, SquaredCost, 0.01)
    layer.W = np.array([[1, 2], [0, -1]])
    layer.b = np.array([[1, 0]]).T

    input = np.array([[1, -1]]).T
    expected_out = np.array([[0, 1]]).T
    assert (layer.forward_pass(input) == expected_out).all()
