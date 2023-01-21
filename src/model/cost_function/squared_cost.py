from . import CostFunction
import numpy as np


class SquaredCost(CostFunction):
    @classmethod
    def cost(cls, Y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        return np.square(Y - pred)

    @classmethod
    def deriv(cls, Y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        return pred - Y
