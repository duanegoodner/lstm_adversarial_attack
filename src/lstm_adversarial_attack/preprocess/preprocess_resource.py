from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExportedPreprocessResource:
    path: Path
    data_type: str
