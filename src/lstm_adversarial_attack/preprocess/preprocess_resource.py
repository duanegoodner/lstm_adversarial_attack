from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict


# @dataclass
# class DataResource:
#     path: Path
#     py_object_type: str


@dataclass
class ExportedPreprocessResource:
    """
    Container to hold path and data type of object exported to pickle file
    """
    path: Path
    data_type: str

    def __repr__(self):
        return f"path: {self.path}, data_type: {self.data_type}"

    def to_dict(self) -> dict[str, str]:
        return {
            "path": str(self.path),
            "data_type": self.data_type
        }

