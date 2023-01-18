from src.dataset_loader.kaggle_dataset_loader import KaggleDatasetLoader
from src.dataset.dig_rec_dataset import DigRecDataSet
import logging

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def main():
    loader = KaggleDatasetLoader()
    path = loader.download("digit-recognizer")
    dataset = DigRecDataSet(path, 0.1)
    dataset.train.plot()


if __name__ == "__main__":
    main()
