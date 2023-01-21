from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass
