from dataclasses import dataclass
from pandas.core.frame import DataFrame  # type: ignore
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import logging


@dataclass
class Data:
    X: np.ndarray
    Y: np.ndarray

    def plot(self) -> None:
        logging.info("Plotting not supported")


class Dataset(ABC):
    train: Data
    test: Data

    @abstractmethod
    def __init__(self, path: Path, test_split: float) -> None:
        pass
