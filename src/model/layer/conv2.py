from . import Layer
from .activation import Activation
from typing import Type, List, Tuple
from src.model.cost_function import CostFunction
from .optimizer import Optimizer
from logging import info

import numpy as np
from src.operations.conv import conv2d, conv2d_sum


class Conv2D(Layer):
    W: np.ndarray
    b: np.ndarray

    def __init__(
        self,
        width: int,
        depth: int,
        activation: Type[Activation],
        regulizer: Type[CostFunction],
        reg_lambda: float,
        optimizer: Optimizer,
    ) -> None:
        self.width = width
        self.depth = depth
        self.activation = activation
        self.regulizer = regulizer
        self.reg_lambda = reg_lambda
        self.optimizer = optimizer

    def random_init(self, input_shape: List[int]):
        self.input_shape = input_shape
        output_shape = input_shape.copy()
        output_shape[2] = self.depth
        self.output_shape = output_shape

        w_scale = np.sqrt(2.0 / self.width)
        self.W = np.random.normal(
            scale=w_scale, size=(self.width, self.width, input_shape[2], self.depth)
        )
        self.b = np.ones(self.depth) * 0.01

    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        x = self.pad(x)
        self.A = x

        res = conv2d(x, self.W)
        for d in range(self.depth):
            res[:, :, d, :] += self.b[d]

        self.Z = res
        return self.activation.apply(res)

    def back_propagation(self, dZ: np.ndarray) -> np.ndarray:
        dZ = np.multiply(dZ, self.activation.derivative(self.Z))

        m = dZ.shape[-1]

        dW = np.zeros(shape=self.W.shape)
        db = np.zeros(shape=self.b.shape)

        for wd in range(self.W.shape[3]):
            for imd in range(self.W.shape[2]):
                res = conv2d_sum(self.A[:, :, imd, :], dZ[:, :, wd, :])
                dW[:, :, imd, wd] = res
            bsum = np.sum(dZ[:, :, wd, :])
            db[wd] = bsum

        dW = dW / m
        db = db / m

        dZ = self.pad(dZ)
        reg_W = self.reg_lambda * self.regulizer.deriv(self.W)

        W_t = self.W.copy()
        self.optimizer.step(self.W, self.b, dW, db, reg_W)

        W_t = np.transpose(W_t, [0, 1, 3, 2])
        W_t = rotate_180(W_t)

        return conv2d(dZ, W_t)

    def print(self) -> None:
        info("--Conv2D Layer--")
        info(f"width {self.width}")
        info(f"depth {self.depth}")
        super().print()

    def pad(self, A: np.ndarray):
        pad_size = self.width // 2
        return np.pad(
            A, ((pad_size, pad_size), (pad_size, pad_size), (0, 0), (0, 0)), "constant"
        )


def rotate_180(A: np.ndarray) -> np.ndarray:
    return A[::-1, ::-1, :]
