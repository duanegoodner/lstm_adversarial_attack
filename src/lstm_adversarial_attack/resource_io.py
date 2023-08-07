# TODO Change to dill and use dill.dump / dill.load syntax.
# TODO Consider removing this module. May be overkill.
import json
from functools import cached_property

import msgspec
import numpy as np
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


class _MsgspecIO:
    @staticmethod
    def to_json(resource: object, path: Path):
        encoded_output = msgspec.json.encode(obj=resource)
        with path.open(mode="wb") as out_file:
            out_file.write(encoded_output)

    @staticmethod
    def from_json(path: Path) -> dict:
        with path.open(mode="rb") as in_file:
            imported_encoded_data = in_file.read()
        return msgspec.json.decode(imported_encoded_data)

class _ListOfArraysIO:

    @staticmethod
    def list_of_np_to_json(resource: list[np.array], path: Path):
        unique_dtypes = np.unique([item.dtype for item in resource])
        assert len(unique_dtypes) == 1
        dtype = unique_dtypes.item()
        list_of_lists = [item.tolist() for item in resource]
        output_dict = {"dtype": dtype, "data": list_of_lists}
        encoded_output = msgspec.json.encode(output_dict)
        with path.open(mode="wb") as out_file:
            out_file.write(encoded_output)

    @staticmethod
    def json_to_list_of_np(path: Path) -> list[np.array]:
        with path.open(mode="rb") as in_file:
            imported_encoded_data = in_file.read()
        imported_dict = msgspec.json.decode(imported_encoded_data)
        list_of_arrays = [
            np.array(item, dtype=imported_dict["dtype"])
            for item in imported_dict["data"]
        ]
        return list_of_arrays


def list_of_np_to_json(resource: list[np.array], path: Path):
    _ListOfArraysIO.list_of_np_to_json(resource=resource, path=path)


def json_to_list_of_np(path: Path) -> list[np.array]:
    return _ListOfArraysIO.json_to_list_of_np(path=path)


class _JsonReadyIO:
    @staticmethod
    def to_json(resource: object, path: Path):
        encoded_output = msgspec.json.encode(obj=resource)
        with path.open(mode="wb") as out_file:
            out_file.write(encoded_output)


class _FullAdmissionDataIO:

    @staticmethod
    def enc_hook(obj: Any) -> Any:
        if isinstance(obj, pd.Timestamp):
            return obj.to_pydatetime()
        if isinstance(obj, pd.DataFrame):
            return obj.to_numpy().tolist()
        if isinstance(obj, np.datetime64):
            return obj.astype(datetime)
        if pd.isna(obj):
            return None
        else:
            raise NotImplementedError(
                f"Encoder does not support objects of type {type(obj)}"
            )

    @cached_property
    def encoder(self) -> msgspec.json.Encoder:
        return msgspec.json.Encoder(enc_hook=self.enc_hook)


FULL_ADMISSION_DATA_IO = _FullAdmissionDataIO()
FULL_ADMISSION_DATA_ENCODER = FULL_ADMISSION_DATA_IO.encoder


def export_full_admission_data(obj: Any, path: Path):
    encoded_object = FULL_ADMISSION_DATA_ENCODER.encode(obj)
    with path.open(mode="wb") as out_file:
        out_file.write(encoded_object)






