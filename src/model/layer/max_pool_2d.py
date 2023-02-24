from . import Layer
import numpy as np
from numpy.lib.stride_tricks import as_strided
from typing import List
import logging


class MaxPool2D(Layer):
    A: np.ndarray

    def __init__(self, scale: int) -> None:
        self.scale = scale

    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        h, w, d, n = x.shape
        assert h % self.scale == 0
        assert w % self.scale == 0

        h_out = h // self.scale
        w_out = w // self.scale

        h_stride = x.strides[0]
        w_stride = x.strides[1]
        d_stride = x.strides[2]
        n_stride = x.strides[3]

        shapes = (h_out, w_out, d, n, self.scale, self.scale)
        strides = (
            h_stride * self.scale,
            w_stride * self.scale,
            d_stride,
            n_stride,
            h_stride,
            w_stride,
        )

        patches = as_strided(x, shape=shapes, strides=strides)

        out: np.ndarray = np.max(patches, axis=(4, 5))

        maxs = out.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        self.A = np.equal(x, maxs).astype(int)

        return out

    def back_propagation(self, dZ: np.ndarray) -> np.ndarray:
        dZ = dZ.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        return np.multiply(dZ, self.A)

    def print(self) -> None:
        logging.info(f"MaxPool2D with scale {self.scale}")

    def random_init(self, input_shape: List[int]):
        output_shape = input_shape.copy()
        output_shape[0] = output_shape[0] // self.scale
        output_shape[1] = output_shape[1] // self.scale
        self.output_shape = output_shape
