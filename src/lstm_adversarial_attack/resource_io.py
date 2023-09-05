# TODO Change to dill and use dill.dump / dill.load syntax.
# TODO Consider removing this module. May be overkill.
from datetime import datetime
from enum import Enum, auto
from functools import cached_property
from pathlib import Path
from typing import Any, Type

import dill
import msgspec
import pandas as pd

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
    filename = f"{prefix}{datetime.now()}{suffix}.{file_extension}".replace(
        " ", "_"
    ).replace(":", "_")
    return parent_path / filename


class ResourceType(Enum):
    CSV = auto()
    PICKLE = auto()


class ResourceImporter:
    _supported_file_types = {
        ".pickle": ResourceType.PICKLE,
    }

    @staticmethod
    def _validate_path(path: Path, file_type: str):
        assert path.exists()
        file_extension = f".{path.name.split('.')[-1]}"
        assert file_type == file_extension

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


class _FeatherIO:

    @staticmethod
    def df_to_feather(df: pd.DataFrame, path: Path):
        assert "index" not in df.columns
        df_for_export = df.reset_index(inplace=False)
        df_for_export.to_feather(path=path)

    @staticmethod
    def feather_to_df(path: Path) -> pd.DataFrame:
        df_with_default_index = pd.read_feather(path=path)
        assert "index" in df_with_default_index.columns
        df = df_with_default_index.set_index("index")
        return df


def df_to_feather(df: pd.DataFrame, path: Path):
    _FeatherIO.df_to_feather(df=df, path=path)


def feather_to_df(path: Path) -> pd.DataFrame:
    return _FeatherIO.feather_to_df(path=path)


class PythonReadyJsonReader:
    @cached_property
    def decoder(self) -> msgspec.json.Decoder:
        return msgspec.json.Decoder()

    def decode(self, obj: bytes) -> Any:
        return self.decoder.decode(obj)

    def import_object(self, path: Path) -> Any:
        with path.open(mode="rb") as in_file:
            encoded_object = in_file.read()
        return self.decode(encoded_object)


PYTHON_READY_JSON_READER = PythonReadyJsonReader()


def import_python_ready_json(path: Path) -> Any:
    return PYTHON_READY_JSON_READER.import_object(path=path)
