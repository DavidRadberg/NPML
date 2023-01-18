
from src.dataset_loader.kaggle_dataset_loader import KaggleDatasetLoader
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def main():
    loader = KaggleDatasetLoader()
    dataset = loader.download("digit-recognizer")


if __name__ == "__main__":
    main()