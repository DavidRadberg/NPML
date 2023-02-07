from . import Optimizer
from numpy import clip, ndarray as array


class LinearOptimizer(Optimizer):
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def step(self, W: array, b: array, dW: array, db: array, reg_W) -> None:
        dW = clip(dW, -0.5, 0.5)
        db = clip(db, -0.5, 0.5)
        W += -self.learning_rate * dW - reg_W
        b += -self.learning_rate * db
