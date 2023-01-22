from . import Activation
import numpy as np


class SoftMax(Activation):
    @classmethod
    def apply(cls, Z: np.ndarray) -> np.ndarray:
        Z = Z - Z.max()
        exp = np.exp(Z)
        return exp / np.sum(exp, 0)

    @classmethod
    def derivative(cls, Z: np.ndarray) -> np.ndarray:
        return np.array(1)
