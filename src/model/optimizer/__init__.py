from abc import ABC, abstractmethod
from numpy import ndarray as array


class Optimizer(ABC):
    @abstractmethod
    def step(self, W: array, b: array, dW: array, db: array) -> None:
        pass
