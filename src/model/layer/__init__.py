from abc import ABC, abstractmethod
import numpy as np
import logging
from .activation import Activation
from typing import Type


class Layer(ABC):
    input_size: int
    output_size: int
    activation: Type[Activation]

    def __init__(self, output_size: int, activation: Type[Activation]) -> None:
        self.output_size = output_size
        self.activation = activation

    def random_init(self, intput_size: int):
        self.input_size = intput_size

    @abstractmethod
    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def back_propagation(self, dz: np.ndarray, alpha: float) -> np.ndarray:
        pass

    def print(self) -> None:
        logging.info("Layer Base Class")
