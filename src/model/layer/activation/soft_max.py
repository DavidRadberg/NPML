from . import Activation
import numpy as np


class SoftMax(Activation):
    @classmethod
    def apply(cls, Z: np.ndarray) -> np.ndarray:
        norm = Z - Z.max(axis=0).reshape(-1, 1).T
        exps = np.exp(norm)
        return exps / sum(exps)

    @classmethod
    def derivative(cls, Z: np.ndarray) -> np.ndarray:
        return np.array(1.0)
