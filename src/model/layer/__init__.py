from abc import ABC, abstractmethod
import numpy as np
from .activation import Activation
from typing import Type, List
from src.model.cost_function import CostFunction
from functools import reduce


class Layer(ABC):
    input_shape: List[int]
    output_shape: List[int]
    activation: Type[Activation]
    regulizer: Type[CostFunction]
    reg_lambda: float

    @abstractmethod
    def random_init(self, input_shape: List[int]):
        pass

    def get_input_size(self) -> int:
        return reduce(lambda x, y: x * y, self.input_shape)

    def get_output_size(self) -> int:
        return reduce(lambda x, y: x * y, self.output_shape)

    @abstractmethod
    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def back_propagation(self, dZ: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def print(self) -> None:
        pass
