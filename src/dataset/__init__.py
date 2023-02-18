from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import logging
import random
from typing import Tuple, List


@dataclass
class Data:
    X: np.ndarray
    Y: np.ndarray
    i: int = 0

    def plot(self) -> None:
        logging.info("Plotting not supported")

    def get_example(self) -> Tuple[np.ndarray, np.ndarray]:
        idx = random.randint(0, self.X.shape[1] - 1)
        x = np.reshape(self.X.T[idx], (2, -1))
        y = np.reshape(self.Y.T[idx], (2, -1))
        return x, y

    def get_batch(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        m = self.X.shape[-1]
        self.i = self.i % m

        if self.i + size > m:
            self.i = m - size - 1

        out = (self.X.T[self.i : self.i + size].T, self.Y.T[self.i : self.i + size].T)
        self.i += size
        return out


class Dataset(ABC):
    train: Data
    test: Data

    @abstractmethod
    def __init__(self, path: Path, test_split: float) -> None:
        pass

    @abstractmethod
    def evaluate(self, Y: np.ndarray) -> float:
        pass

    def input_shape(self) -> List[int]:
        return list(self.train.X.shape[0:-1])

    def output_shape(self) -> List[int]:
        return list(self.train.Y.shape[0:-1])

    def get_training_batch(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.train.get_batch(size)
