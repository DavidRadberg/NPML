from abc import ABCMeta, abstractmethod
import numpy as np


class Activation(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def apply(cls, x: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    @abstractmethod
    def derivative(cls, x: np.ndarray) -> np.ndarray:
        pass
