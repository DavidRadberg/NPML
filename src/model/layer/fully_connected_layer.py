from . import Layer
import numpy as np


class FullyConnectedLayer(Layer):
    W: np.ndarray
    b: np.ndarray
    A: np.ndarray
    Z: np.ndarray

    def random_init(self, input_size: int):
        self.W = np.random.rand(self.output_size, input_size) - 0.5
        self.b = np.random.rand(self.output_size, 1) - 0.5

        super().random_init(input_size)

    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        self.A = x
        self.Z = self.W.dot(x) + self.b
        return self.activation.apply(self.Z)

    def back_propagation(self, dZ: np.ndarray, alpha: float) -> np.ndarray:
        dZ = dZ * self.activation.derivative(self.Z)
        m = dZ.shape[1]
        dW: np.ndarray = 1 / m * dZ.dot(self.A.T)
        db: np.ndarray = 1 / m * np.sum(dZ)
        self.W = self.W - dW * alpha
        self.b = self.b - db * alpha

        return self.W.T.dot(dZ)
