from . import Activation
import numpy as np


class LinearActivation(Activation):
    @classmethod
    def apply(cls, Z: np.ndarray) -> np.ndarray:
        return Z

    @classmethod
    def derivative(cls, Z: np.ndarray) -> np.ndarray:
        return np.array(1.0)
