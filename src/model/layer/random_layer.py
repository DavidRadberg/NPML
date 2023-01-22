from . import Layer
import numpy as np


class RandomLayer(Layer):
    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        return np.random.uniform(
            low=-1.0, high=1.0, size=(self.output_size, x.shape[1])
        )

    def back_propagation(self, dz: np.ndarray, alpha: float) -> np.ndarray:
        return np.random.uniform(
            low=-1.0, high=1.0, size=(self.input_size, dz.shape[1])
        )
