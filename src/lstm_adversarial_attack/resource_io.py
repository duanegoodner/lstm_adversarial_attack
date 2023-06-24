# TODO Change to dill and use dill.dump / dill.load syntax.
# TODO Consider removing this module. May be overkill.
import dill
import pandas as pd
import sys
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
import lstm_adversarial_attack.custom_unpickler as cu


def create_timestamped_dir(parent_path: Path) -> Path:
    dirname = f"{datetime.now()}".replace(" ", "_").replace(":", "_")
    new_dir_path = parent_path / dirname
    assert not new_dir_path.exists()
    new_dir_path.mkdir()
    return new_dir_path


def create_timestamped_filepath(
    parent_path: Path, file_extension: str, prefix: str = "", suffix: str = ""
):
    filename = f"{prefix}{datetime.now()}{suffix}.{file_extension}".replace(" ", "_").replace(
        ":", "_"
    )
    return parent_path / filename


class ResourceType(Enum):
    CSV = auto()
    PICKLE = auto()


class ResourceImporter:
    _supported_file_types = {
        ".csv": ResourceType.CSV,
        ".pickle": ResourceType.PICKLE,
    }

    # def _validate_path(self, path: Path) -> ResourceType:
    #     assert path.exists()
    #     file_extension = f".{path.name.split('.')[-1]}"
    #     file_type = self._supported_file_types.get(file_extension)
    #     assert file_type is not None
    #     return file_type

    @staticmethod
    def _validate_path(path: Path, file_type: str):
        assert path.exists()
        file_extension = f".{path.name.split('.')[-1]}"
        assert file_type == file_extension

    def import_csv(self, path: Path) -> pd.DataFrame:
        self._validate_path(path=path, file_type=".csv")
        return pd.read_csv(path)

    def import_pickle_to_df(self, path: Path) -> pd.DataFrame:
        self._validate_path(path=path, file_type=".pickle")
        with path.open(mode="rb") as p:
            result = cu.load(p)
        return result

    def import_pickle_to_object(self, path: Path) -> object:
        self._validate_path(path=path, file_type=".pickle")
        with path.open(mode="rb") as p:
            result = cu.load(p)
        return result

    def import_pickle_to_list(self, path: Path) -> list:
        self._validate_path(path=path, file_type=".pickle")
        with path.open(mode="rb") as p:
            result = cu.load(p)
        return result


class ResourceExporter:
    _supported_file_types = [".pickle"]

    def export(self, resource: object, path: Path):
        self._validate_path(path=path)
        with path.open(mode="wb") as p:
            dill.dump(obj=resource, file=p)

    def _validate_path(self, path: Path):
        assert f".{path.name.split('.')[-1]}" in self._supported_file_types
