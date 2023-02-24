from . import Layer
import numpy as np
from logging import info
from .activation import Activation
from typing import Type, List
from src.model.cost_function import CostFunction
from .optimizer import Optimizer


class FullyConnectedLayer(Layer):
    W: np.ndarray
    b: np.ndarray
    A: np.ndarray
    Z: np.ndarray

    def __init__(
        self,
        output_shape: List[int],
        activation: Type[Activation],
        regulizer: Type[CostFunction],
        reg_lambda: float,
        optimizer: Optimizer,
    ) -> None:
        self.output_shape = output_shape
        self.activation = activation
        self.regulizer = regulizer
        self.reg_lambda = reg_lambda
        self.optimizer = optimizer

    def random_init(self, input_shape: List[int]):
        self.input_shape = input_shape
        input_size = self.get_input_size()
        output_size = self.get_output_size()
        scale = np.sqrt(2.0 / input_size)

        self.W = np.random.normal(scale=scale, size=(output_size, input_size))
        self.b = np.random.normal(scale=scale, size=(output_size, 1))

    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        x = flatten(x)

        self.A = x
        self.Z = self.W.dot(x) + self.b
        return self.activation.apply(self.Z)

    def back_propagation(self, dZ: np.ndarray) -> np.ndarray:
        dZ = flatten(dZ)

        dZ = np.multiply(dZ, self.activation.derivative(self.Z))
        m = dZ.shape[1]
        reg_W = self.reg_lambda * self.regulizer.deriv(self.W)
        dW: np.ndarray = dZ.dot(self.A.T) / m
        db: np.ndarray = np.sum(dZ, axis=1).reshape(-1, 1) / m

        dZ = self.W.T.dot(dZ)
        self.optimizer.step(self.W, self.b, dW, db, reg_W)

        shape = self.input_shape.copy()
        shape.append(-1)
        return dZ.reshape(shape)

    def print(self) -> None:
        info(f"--Fully Connected Layer--")
        super().print()


def flatten(A: np.ndarray) -> np.ndarray:
    return A.reshape((-1, A.shape[-1]))
