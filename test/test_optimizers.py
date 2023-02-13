from src.model.layer.fully_connected_layer import FullyConnectedLayer
from src.model.layer.optimizer.linear_optimizer import LinearOptimizer
from src.model.layer.optimizer.adam import Adam
from src.model.layer.activation.linear_activation import LinearActivation
from src.model.cost_function.squared_cost import SquaredCost

import numpy as np


def test_linear_optimizer() -> None:
    layer = FullyConnectedLayer(
        2, LinearActivation, SquaredCost, 0.0, LinearOptimizer(0.1)
    )
    layer.random_init(1)

    W_org = np.copy(layer.W)

    layer.forward_pass(np.array([[1.0]]))
    layer.back_propagation(np.array([[1.0, -1.0]]).T)

    assert layer.W[0][0] < W_org[0][0]
    assert layer.W[1][0] > W_org[1][0]


def test_adam() -> None:
    layer = FullyConnectedLayer(2, LinearActivation, SquaredCost, 0.0, Adam())
    layer.random_init(1)

    W_org = np.copy(layer.W)

    layer.forward_pass(np.array([[1.0]]))
    layer.back_propagation(np.array([[1.0, -1.0]]).T)

    assert layer.W[0][0] < W_org[0][0]
    assert layer.W[1][0] > W_org[1][0]
