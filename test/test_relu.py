from src.model.layer.activation.relu import Relu
import numpy as np


def test_relu():
    input = np.array([[-0.2, -0.4, 1.0], [2.0, -2.0, 1.2]])
    expected_output = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 1.2]])
    expected_deriv = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])

    output = Relu.apply(input)
    deriv = Relu.derivative(input)

    assert np.all(deriv == expected_deriv)
    assert np.all(output == expected_output)
