from abc import ABCMeta, abstractmethod
from numpy import ndarray


class CostFunction(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def cost(cls, Y: ndarray, pred: ndarray) -> ndarray:
        pass

    @classmethod
    @abstractmethod
    def deriv(cls, Y: ndarray, pred: ndarray) -> ndarray:
        pass
