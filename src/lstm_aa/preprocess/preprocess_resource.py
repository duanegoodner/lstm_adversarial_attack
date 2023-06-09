from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataResource:
    path: Path
    py_object_type: str


@dataclass
class ExportedPreprocessResource:
    path: Path
    data_type: str
