# TODO Change to dill and use dill.dump / dill.load syntax.
# TODO Consider removing this module. May be overkill.
import json

import dill
import pandas as pd
import sys
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any
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


class DataFrameIO:
    @staticmethod
    def df_to_json(df: pd.DataFrame, path: Path):
        assert f".{path.name.split('.')[-1]}" == ".json"
        dtypes = df.dtypes.apply(lambda x: x.name).to_dict()
        data_json_str = df.to_json()
        typed_df_dict = {"dtypes": dtypes, "data": data_json_str}
        with path.open(mode="w") as out_file:
            json.dump(obj=typed_df_dict, fp=out_file)

    @staticmethod
    def json_to_df(path: Path) -> pd.DataFrame:
        with path.open(mode="r") as in_file:
            typed_df_dict = json.load(fp=in_file)
        return pd.read_json(typed_df_dict["data"]).astype(
            typed_df_dict["dtypes"]
        )

    @staticmethod
    def df_to_pickle(resource: pd.DataFrame, path: Path):
        assert f".{path.name.split('.')[-1]}" == ".pickle"
        with path.open(mode="wb") as p:
            dill.dump(obj=resource, file=p)

    @staticmethod
    def pickle_to_df(path: Path) -> pd.DataFrame:
        assert f".{path.name.split('.')[-1]}" == ".pickle"
        with path.open(mode="rb") as p:
            result = cu.load(p)
        return result


def df_to_json(resource: pd.DataFrame, path: Path):
    DataFrameIO.df_to_json(df=resource, path=path)

def df_to_pickle(resource: pd.DataFrame, path: Path):
    DataFrameIO.df_to_pickle(resource=resource, path=path)

def json_to_df(path: Path) -> pd.DataFrame:
    return DataFrameIO.json_to_df(path=path)

def pickle_to_df(path: Path) -> pd.DataFrame:
    return DataFrameIO.pickle_to_df(path=path)

def import_json(path: Path) -> dict[str, Any]:
    with path.open(mode="r") as in_file:
        input_dict = json.load(in_file)
    return input_dict

def export_to_pickle(resource: object, path: Path):
    assert f".{path.name.split('.')[-1]}" == ".pickle"
    with path.open(mode="wb") as p:
        dill.dump(obj=resource, file=p)

def convert_posix_paths_to_strings(data):
    if isinstance(data, dict):
        return {key: convert_posix_paths_to_strings(value) for key, value
                in data.items()}
    elif isinstance(data, list):
        return [convert_posix_paths_to_strings(item) for item in data]
    elif isinstance(data, Path):  # Check if the value is a PosixPath
        return str(data)
    else:
        return data


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


