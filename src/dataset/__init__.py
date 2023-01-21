from dataclasses import dataclass
from pandas.core.frame import DataFrame  # type: ignore
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import logging
import random
from typing import Tuple


@dataclass
class Data:
    X: np.ndarray
    Y: np.ndarray

    def plot(self) -> None:
        logging.info("Plotting not supported")

    def get_example(self) -> Tuple[np.ndarray, np.ndarray]:
        idx = random.randint(0, self.X.shape[1] - 1)
        x = np.reshape(self.X.T[idx], (2, -1))
        y = np.reshape(self.Y.T[idx], (2, -1))
        return x, y


class Dataset(ABC):
    train: Data
    test: Data

    @abstractmethod
    def __init__(self, path: Path, test_split: float) -> None:
        pass
