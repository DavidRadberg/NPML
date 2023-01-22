from . import Activation
import numpy as np


class Relu(Activation):
    @classmethod
    def apply(cls, Z: np.ndarray) -> np.ndarray:
        return np.maximum(Z, 0)

    @classmethod
    def derivative(cls, Z: np.ndarray) -> np.ndarray:
        return Z > 0
