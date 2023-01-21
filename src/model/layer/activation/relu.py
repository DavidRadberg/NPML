from . import Activation
import numpy as np


class Relu(Activation):
    @classmethod
    def apply(cls, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    @classmethod
    def derivative(cls, x: np.ndarray) -> np.ndarray:
        return x > 0.0
