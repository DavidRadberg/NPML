from abc import ABC
from abc import abstractmethod
from pathlib import Path


class DatasetLoader(ABC):
    @abstractmethod
    def download(self, name: str) -> Path:
        pass
