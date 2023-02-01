from src.model.layer.activation.relu import Relu
from src.model.layer.activation.soft_max import SoftMax
import numpy as np


def test_relu():
    input = np.array([[-0.2, -0.4, 1.0], [2.0, -2.0, 1.2]])
    expected_output = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 1.2]])
    expected_deriv = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])

    output = Relu.apply(input)
    deriv = Relu.derivative(input)

    assert np.all(deriv == expected_deriv)
    assert np.all(output == expected_output)


def test_softmax():
    input = np.array([[0.0, 1.0, 2.0], [200000000, -20, 1.2], [2.0, 2.0, 2.0]]).T
    expected_sum = np.array([1.0, 1.0, 1.0])

    s1 = np.exp(0) + np.exp(1) + np.exp(2)
    expected_output = np.array(
        [[1 / s1, np.e / s1, np.exp(2) / s1], [1.0, 0.0, 0.0], [1 / 3, 1 / 3, 1 / 3]]
    ).T
    output = SoftMax.apply(input)

    out_sum = np.sum(output, 0)
    assert np.allclose(output, expected_output)
    assert np.allclose(out_sum, expected_sum)
    assert SoftMax.derivative(input) == 1.0
