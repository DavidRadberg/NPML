from .layer import Layer
from typing import List
import numpy as np


class Model:
    layers: List[Layer] = []

    def __init__(self, input_size: int) -> None:
        self.input_size = input_size

    def get_output_size(self):
        if len(self.layers) > 0:
            return self.layers[-1].output_size
        else:
            return self.input_size

    def add_layer(self, layer: Layer):
        layer.random_init(self.get_output_size())
        self.layers.append(layer)

    def run(self, X: np.ndarray):
        for layer in self.layers:
            X = layer.forward_pass(X)
        return X

    def gradiend_descent(self, X: np.ndarray, Y: np.ndarray, alpha: float):
        prediction = self.run(X)
