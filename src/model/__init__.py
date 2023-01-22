from .layer import Layer
from typing import List
import numpy as np
from .cost_function import CostFunction
from typing import Type


class Model:
    layers: List[Layer] = []
    cost_function: Type[CostFunction]

    def __init__(self, input_size: int, cost_function: Type[CostFunction]) -> None:
        self.input_size = input_size
        self.cost_function = cost_function

    def get_output_size(self):
        if len(self.layers) > 0:
            return self.layers[-1].output_size
        else:
            return self.input_size

    def add_layer(self, layer: Layer):
        layer.random_init(self.get_output_size())
        self.layers.append(layer)

    def run(self, X: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            X = layer.forward_pass(X)
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        Y = self.run(X)
        pred = np.zeros(Y.shape)
        idx = np.argmax(Y, 0)
        for p, i in zip(pred.T, idx):
            p[i] = 1
        return pred

    def gradiend_descent(self, X: np.ndarray, Y: np.ndarray, alpha: float):
        prediction = self.run(X)
        dz = self.cost_function.deriv(Y, prediction)
        for layer in reversed(self.layers):
            dz = layer.back_propagation(dz, alpha)
