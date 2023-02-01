from . import Dataset, Data
from pathlib import Path

import numpy as np


def create_mock_data(size: int) -> Data:
    x1 = np.random.random(size)
    x2 = np.random.random(size)
    x3 = np.random.random(size)

    y1 = x1 > x2
    y2 = x2 > x1
    return Data(np.array([x1.T, x2.T, x3.T]), np.array([y1.T, y2.T]))


class TrivialDataset(Dataset):
    def __init__(self, path: Path, test_split: float) -> None:
        self.train = create_mock_data(10000)
        self.test = create_mock_data(100)

    def evaluate(self, Y: np.ndarray) -> float:
        prediction = np.zeros(Y.shape)
        idx = np.argmax(Y, 0)
        for p, i in zip(prediction.T, idx):
            p[i] = 1

        n_correct = 0
        for y, p in zip(self.test.Y.T, prediction.T):
            if np.all(y == p):
                n_correct += 1
        return n_correct / len(prediction.T)
