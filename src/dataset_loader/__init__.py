from abc import ABC
from abc import abstractmethod
from src.dataset import Dataset

class DatasetLoader(ABC):
    @abstractmethod
    def download(self, name: str) -> Dataset:
        pass
