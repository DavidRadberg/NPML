from . import CostFunction
import numpy as np


class SquaredCost(CostFunction):
    @classmethod
    def cost(cls, delta: np.ndarray) -> np.ndarray:
        return np.square(delta) / 2

    @classmethod
    def deriv(cls, delta: np.ndarray) -> np.ndarray:
        return delta
