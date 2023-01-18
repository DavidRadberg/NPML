
from dataclasses import dataclass
from pandas.core.frame import DataFrame #type: ignore

@dataclass
class Dataset:
    train: DataFrame
    test: DataFrame