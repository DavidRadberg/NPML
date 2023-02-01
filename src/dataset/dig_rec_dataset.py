from . import Data, Dataset
from pathlib import Path
from pandas import read_csv  # type: ignore
import numpy as np
from matplotlib.pyplot import imshow, show, title  # type: ignore
import logging


NUM_CLASSES = 10
IMG_SIZE = 28


class DigRecData(Data):
    def plot(self) -> None:
        X, Y = self.get_example()
        img = np.reshape(X, (IMG_SIZE, IMG_SIZE))
        imshow(img, cmap="gray")
        logging.info(f"Correct label is {np.argmax(Y)}")
        show()


def label_to_y(label: int) -> np.ndarray:
    y = np.zeros(NUM_CLASSES)
    y[label] = 1
    return y


def dig_rec_data_factory(data: np.ndarray) -> DigRecData:
    data = data.T
    Y = np.array([label_to_y(d) for d in data[0]]).T
    X = data[1:]
    X = X / float(X.max())
    return DigRecData(X, Y)


class DigRecDataSet(Dataset):
    def __init__(self, path: Path, test_split: float) -> None:
        data = np.array(read_csv(path))
        np.random.shuffle(data)
        m_test = max(int(test_split * data.shape[0]), 1)
        self.test = dig_rec_data_factory(data[0:m_test])
        self.train = dig_rec_data_factory(data[m_test:])

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
