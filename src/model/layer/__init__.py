from abc import ABC, abstractmethod
import numpy as np
import logging
from .activation import Activation
from typing import Type
from src.model.cost_function import CostFunction
from .optimizer import Optimizer


class Layer(ABC):
    input_size: int
    output_size: int
    activation: Type[Activation]
    regulizer: Type[CostFunction]
    reg_lambda: float

    def __init__(
        self,
        output_size: int,
        activation: Type[Activation],
        regulizer: Type[CostFunction],
        reg_lambda: float,
        optimizer: Optimizer,
    ) -> None:
        self.output_size = output_size
        self.activation = activation
        self.regulizer = regulizer
        self.reg_lambda = reg_lambda
        self.optimizer = optimizer

    def random_init(self, intput_size: int):
        self.input_size = intput_size

    @abstractmethod
    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def back_propagation(self, dZ: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def print(self) -> None:
        pass
