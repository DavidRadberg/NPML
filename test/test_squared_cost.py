from src.model.cost_function.squared_cost import SquaredCost
import numpy as np


def test_squared_loss():
    Y = np.array([[-1.0, 2.0], [3.0, -2.0]])
    pred = np.array([[1.0, 2.0], [1.0, -1.0]])

    expected_cost = 0.5 * np.array([[4.0, 0.0], [4.0, 1.0]])
    expected_deriv = np.array([[2.0, 0.0], [-2.0, 1.0]])

    cost = SquaredCost.cost(pred - Y)
    deriv = SquaredCost.deriv(pred - Y)

    assert np.all(cost == expected_cost)
    assert np.all(deriv == expected_deriv)
