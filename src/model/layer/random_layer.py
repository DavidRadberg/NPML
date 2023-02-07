from . import Layer
import numpy as np


class RandomLayer(Layer):
    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        return np.random.uniform(
            low=-1.0, high=1.0, size=(self.output_size, x.shape[1])
        )

    def back_propagation(self, dZ: np.ndarray) -> np.ndarray:
        return np.random.uniform(
            low=-1.0, high=1.0, size=(self.input_size, dZ.shape[1])
        )

    def print(self):
        print("Random Layer")
