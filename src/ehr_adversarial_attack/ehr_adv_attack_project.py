from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataResource:
    path: Path
    py_object_type: str


class MimiciiiDatabaseInterface(ABC):
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def run_sql_queries(self, sql_queries: list[Path]) -> list[Path]:
        pass

    @abstractmethod
    def close_connection(self):
        pass


class DataPreprocessor(ABC):
    @abstractmethod
    def preprocess(self) -> dict[str, DataResource]:
        pass


class ModelTrainer(ABC):
    @abstractmethod
    def train_model(self) -> dict[str, DataResource]:
        pass


class EHRAdvAttackProject:
    def __init__(
        self,
        db_interface: MimiciiiDatabaseInterface,
        preprocessor: DataPreprocessor,
    ):
        self._db_interface = db_interface
        self._preprocessor = preprocessor
