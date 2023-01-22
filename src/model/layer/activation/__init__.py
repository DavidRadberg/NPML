from abc import ABCMeta, abstractmethod
import numpy as np


class Activation(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def apply(cls, Z: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    @abstractmethod
    def derivative(cls, Z: np.ndarray) -> np.ndarray:
        pass
