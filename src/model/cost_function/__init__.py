from abc import ABCMeta, abstractmethod
from numpy import ndarray


class CostFunction(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def cost(cls, delta: ndarray) -> ndarray:
        pass

    @classmethod
    @abstractmethod
    def deriv(cls, delta: ndarray) -> ndarray:
        pass
