from src.model.layer.random_layer import RandomLayer
from src.model.layer.activation.relu import Relu
from src.model.layer.fully_connected_layer import FullyConnectedLayer
from src.model.layer.conv2 import Conv2D
from src.model.cost_function.squared_cost import SquaredCost
import numpy as np
from src.model.layer.optimizer.linear_optimizer import LinearOptimizer


def test_shape():
    layer = RandomLayer([10])
    layer.random_init([20])
    assert layer.input_shape == [20]
    assert layer.output_shape == [10]


def test_fully_connected_layer():
    layer = FullyConnectedLayer(2, Relu, SquaredCost, 0.01, LinearOptimizer(0.1))
    layer.W = np.array([[1, 2], [0, -1]])
    layer.b = np.array([[1, 0]]).T

    input = np.array([[1, -1]]).T
    expected_out = np.array([[0, 1]]).T
    assert (layer.forward_pass(input) == expected_out).all()


def test_cnn_layer():
    layer = Conv2D(
        width=3,
        depth=5,
        activation=Relu,
        regulizer=SquaredCost,
        reg_lambda=0.01,
        optimizer=LinearOptimizer(0.1),
    )

    input_shape = [10, 14, 3]
    layer.random_init(input_shape)

    assert layer.input_shape == input_shape
    assert layer.output_shape == [10, 14, 5]

    input_data = np.random.uniform(size=[10, 14, 3, 10])
    res = layer.forward_pass(input_data)
    assert res.shape == (10, 14, 5, 10)

    dZ = np.random.uniform(size=(10, 14, 5, 10))
    dZ = layer.back_propagation(dZ)
    assert dZ.shape == (10, 14, 3, 10)
