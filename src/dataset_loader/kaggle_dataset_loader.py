from . import DatasetLoader
from kaggle.api.kaggle_api_extended import KaggleApi # type: ignore
from pathlib import Path
import pandas as pd #type: ignore
from src.dataset import Dataset

import zipfile
import os
import logging


DATA_PATH = "data/"

class KaggleDatasetLoader(DatasetLoader):
    def download(self, name: str) -> Dataset:
        output_path = Path(DATA_PATH + name)
        zip_name = Path(DATA_PATH + name + '.zip')

        if not os.path.exists(output_path):
            logging.info(f"Downloading {name} dataset...")
            api = KaggleApi()
            api.authenticate()
            api.competition_download_files(name, path=DATA_PATH)

            logging.info("Unzipping...")

            with zipfile.ZipFile(zip_name, 'r') as zip_ref:
                zip_ref.extractall(output_path)
            Path.unlink(zip_name)
        else:
            logging.info(f"{name} dataset already exists")


        train_data = pd.read_csv(output_path / 'train.csv')
        test_data = pd.read_csv(output_path / 'test.csv')

        dataset = Dataset(train_data, test_data)
        return dataset



