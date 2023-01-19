from abc import ABC, abstractmethod
import numpy as np
import logging


class Layer(ABC):
    def __init__(self, input_size: int, output_size: int) -> None:
        self.input_size = input_size
        self.output_size = output_size

    @abstractmethod
    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def back_propagation(self, dz: np.ndarray) -> np.ndarray:
        pass

    def print(self) -> None:
        logging.info("Layer Base Class")
