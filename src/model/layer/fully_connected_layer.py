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

    def back_propagation(self, dZ: np.ndarray, alpha: float) -> np.ndarray:
        dZ = np.multiply(dZ, self.activation.derivative(self.Z))
        m = dZ.shape[1]
        reg_W = self.reg_lambda * self.regulizer.deriv(self.W)
        dW: np.ndarray = 1 / m * dZ.dot(self.A.T)
        db: np.ndarray = 1 / m * np.sum(dZ, axis=1).reshape(-1, 1)
        dW = np.clip(dW, -0.5, 0.5)
        db = np.clip(db, -0.5, 0.5)
        dz = self.W.T.dot(dZ)
        self.W = self.W - dW * alpha - reg_W
        self.b = self.b - db * alpha

        return dz

    def print(self) -> None:
        info("Fully connected Layer")
        info(f"Max, min of W is {self.W.max()}, {self.W.min()}")
        info(f"Max, min of b is {self.b.max()}, {self.b.min()}")
