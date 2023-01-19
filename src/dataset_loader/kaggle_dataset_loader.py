from . import DatasetLoader
from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
from pathlib import Path
import pandas as pd  # type: ignore

import zipfile
import os
import logging


class KaggleDatasetLoader(DatasetLoader):
    def download(self, name: str, path: str) -> Path:
        output_path = Path(path + name)
        zip_name = Path(path + name + ".zip")

        if not os.path.exists(output_path):
            logging.info(f"Downloading {name} dataset...")
            api = KaggleApi()
            api.authenticate()
            api.competition_download_files(name, path=path)

            logging.info("Unzipping...")

            with zipfile.ZipFile(zip_name, "r") as zip_ref:
                zip_ref.extractall(output_path)
            Path.unlink(zip_name)
        else:
            logging.info(f"{name} dataset already exists")

        return Path(output_path / "train.csv")
