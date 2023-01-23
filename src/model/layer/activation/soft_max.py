from . import Activation
import numpy as np


class SoftMax(Activation):
    @classmethod
    def apply(cls, Z: np.ndarray) -> np.ndarray:
        Z = Z - np.max(Z, axis=0, keepdims=True)
        numerator = np.exp(Z)
        denominator = np.sum(numerator, axis=0, keepdims=True)
        softmax = numerator / denominator
        return softmax

    @classmethod
    def derivative(cls, Z: np.ndarray) -> np.ndarray:
        return np.array(1)
