from . import Layer
import numpy as np
from typing import List
import numpy as np


class RandomLayer(Layer):
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def random_init(self, input_shape: List[int]):
        self.input_shape = input_shape

    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        shape = self.output_shape.copy()
        shape.append(x.shape[-1])

        return np.random.normal(scale=1.0, size=shape)

    def back_propagation(self, dZ: np.ndarray) -> np.ndarray:
        shape = self.input_shape.copy()
        shape.append(dZ.shape[-1])

        return np.random.normal(scale=1.0, size=shape)

    def print(self):
        print("--Random Layer--")
        super().print()
