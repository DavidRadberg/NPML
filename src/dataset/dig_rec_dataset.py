from . import Data, Dataset
from pathlib import Path
from pandas import read_csv  # type: ignore
import numpy as np
from matplotlib.pyplot import imshow, show, title  # type: ignore
import random


NUM_CLASSES = 10
IMG_SIZE = 28


class DigRecData(Data):
    def plot(self) -> None:
        X, _ = self.get_example()
        img = np.reshape(X, (IMG_SIZE, IMG_SIZE))
        imshow(img, cmap="gray")
        show()


def label_to_y(label: int) -> np.ndarray:
    y = np.zeros(NUM_CLASSES)
    y[label] = 1
    return y


def dig_rec_data_factory(data: np.ndarray) -> DigRecData:
    data = data.T
    Y = np.array([label_to_y(d) for d in data[0]]).T
    X = data[1:]
    return DigRecData(X, Y)


class DigRecDataSet(Dataset):
    def __init__(self, path: Path, test_split: float) -> None:
        data = np.array(read_csv(path))
        np.random.shuffle(data)
        m_test = max(int(test_split * data.shape[0]), 1)
        self.test = dig_rec_data_factory(data[0:m_test])
        self.train = dig_rec_data_factory(data[m_test:])
