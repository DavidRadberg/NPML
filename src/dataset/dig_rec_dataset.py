from . import Data, Dataset
from pathlib import Path
from pandas import read_csv  # type: ignore
import numpy as np
from matplotlib.pyplot import imshow, show  # type: ignore
import random


class DigRecData(Data):
    w = 28
    h = 28

    def plot(self) -> None:
        idx = random.randint(0, self.X.shape[1] - 1)
        img = np.reshape(self.X.T[idx], (self.w, self.h))
        imshow(img, cmap="gray")
        show()


def dig_rec_data_factory(data: np.ndarray) -> DigRecData:
    data = data.T
    Y = data[0]
    X = data[1:]
    return DigRecData(X, Y)


class DigRecDataSet(Dataset):
    def __init__(self, path: Path, test_split: float) -> None:
        data = np.array(read_csv(path))
        m_test = int(test_split * data.shape[0])
        self.test = dig_rec_data_factory(data[0:m_test])
        self.train = dig_rec_data_factory(data[m_test:])
