from . import Layer
import numpy as np
from logging import info


class FullyConnectedLayer(Layer):
    W: np.ndarray
    b: np.ndarray
    A: np.ndarray
    Z: np.ndarray

    def random_init(self, input_size: int):
        scale = np.sqrt(1.0 / input_size)
        self.W = np.random.normal(scale=scale, size=(self.output_size, input_size))
        self.b = np.random.normal(scale=scale, size=(self.output_size, 1))

        super().random_init(input_size)

    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        self.A = x
        self.Z = self.W.dot(x) + self.b
        return self.activation.apply(self.Z)

    def back_propagation(self, dZ: np.ndarray) -> np.ndarray:
        dZ = np.multiply(dZ, self.activation.derivative(self.Z))
        m = dZ.shape[1]
        reg_W = self.reg_lambda * self.regulizer.deriv(self.W)
        dW: np.ndarray = dZ.dot(self.A.T) / m
        db: np.ndarray = np.sum(dZ, axis=1).reshape(-1, 1) / m

        dz = self.W.T.dot(dZ)
        self.optimizer.step(self.W, self.b, dW, db, reg_W)

        return dz

    def print(self) -> None:
        info(f"Fully connected Layer with size {self.input_size}, {self.output_size}")
        info(f"Max, min of W is {self.W.max()}, {self.W.min()}")
        info(f"Max, min of b is {self.b.max()}, {self.b.min()}")
