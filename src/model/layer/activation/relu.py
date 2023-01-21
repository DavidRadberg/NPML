from . import Activation
import numpy as np


class Relu(Activation):
    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return x > 0.0
