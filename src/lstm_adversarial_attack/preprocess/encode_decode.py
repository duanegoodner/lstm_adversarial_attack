from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Type, TypeVar

import msgspec
import numpy as np
import pandas as pd
import torch

import lstm_adversarial_attack.model.model_data_structs as mds
import lstm_adversarial_attack.preprocess.encode_decode_structs as eds
from lstm_adversarial_attack.config import CONFIG_READER

# TODO Consider creating DataWriter base class with encoder abstractmethod and
#  encode() & export() concrete methods

ADMISSION_DATA_JSON_DELIMITER = CONFIG_READER.get_config_value(
    config_key="preprocess.admission_data_json_delimiter"
)


class AdmissionDataWriter:
    def __init__(self, delimiter: str = ADMISSION_DATA_JSON_DELIMITER):
        self._delimiter = delimiter

    @staticmethod
    def enc_hook(obj: Any) -> Any:
        if isinstance(obj, pd.Timestamp):
            return obj.to_pydatetime()
        if isinstance(obj, pd.DataFrame):
            return eds.DecomposedTimeSeries(
                index=obj.index.tolist(),
                time_vals=obj["charttime"].tolist(),
                data=obj.loc[:, obj.columns != "charttime"]
                .to_numpy()
                .tolist(),
            )
        if isinstance(obj, np.datetime64):
            return pd.Timestamp(obj)
        if pd.isna(obj):
            return None
        else:
            raise NotImplementedError(
                f"Encoder does not support objects of type {type(obj)}"
            )

    @cached_property
    def encoder(self) -> msgspec.json.Encoder:
        return msgspec.json.Encoder(enc_hook=self.enc_hook)

    def encode(
        self, full_admission_data_list: list[eds.FullAdmissionData]
    ) -> bytes:
        example_df = full_admission_data_list[0].time_series
        timestamp_col_name = "charttime"
        timestamp_dtype = example_df[timestamp_col_name].dtype.name
        data_cols_names = [
            item for item in example_df.columns if item != timestamp_col_name
        ]
        data_only_df = example_df[data_cols_names]
        data_cols_dtype = np.unique(data_only_df.dtypes).item().name

        header = eds.FullAdmissionDataListHeader(
            timestamp_col_name=timestamp_col_name,
            timestamp_dtype=timestamp_dtype,
            data_cols_names=data_cols_names,
            data_cols_dtype=data_cols_dtype,
        )

        return self.encoder.encode(
            (
                header,
                self._delimiter,
                full_admission_data_list,
            )
        )

    def export(self, obj: Any, path: Path):
        encoded_data = self.encode(obj)
        with path.open(mode="wb") as out_file:
            out_file.write(encoded_data)


def export_admission_data_list(
    admission_data_list: list[eds.FullAdmissionData], path: Path
):
    AdmissionDataWriter().export(obj=admission_data_list, path=path)


class AdmissionDataListReader:
    def __init__(
        self,
        encoded_data: bytes,
        delimiter: str = ADMISSION_DATA_JSON_DELIMITER,
    ):
        self._encoded_data = encoded_data
        self._delimiter = delimiter

    @classmethod
    def from_file(
        cls, path: Path, delimiter: str = ADMISSION_DATA_JSON_DELIMITER
    ):
        with path.open(mode="rb") as in_file:
            encoded_data = in_file.read()
        return cls(encoded_data=encoded_data, delimiter=delimiter)

    @cached_property
    def _header_and_body_bytes(self) -> tuple[bytes, bytes]:
        index = self._encoded_data.find(bytes(self._delimiter, "utf-8"))
        assert index != -1

        header = self._encoded_data[1 : index - 2]
        body = self._encoded_data[index + len(self._delimiter) + 2 : -1]
        return header, body

    @cached_property
    def _header_bytes(self) -> bytes:
        return self._header_and_body_bytes[0]

    @cached_property
    def _body_bytes(self) -> bytes:
        return self._header_and_body_bytes[1]

    @cached_property
    def _header_decoder(self) -> msgspec.json.Decoder:
        return msgspec.json.Decoder(eds.FullAdmissionDataListHeader)

    @cached_property
    def _header(self) -> eds.FullAdmissionDataListHeader:
        return self._header_decoder.decode(self._header_bytes)

    def _body_dec_hook(self, type: Type, obj: Any) -> Any:
        if type is pd.Timestamp:
            return pd.Timestamp(obj)
        if type is pd.DataFrame:
            time_vals = np.array(obj["time_vals"], dtype="datetime64[ns]")
            df = pd.DataFrame(
                np.array(obj["data"], dtype=self._header.data_cols_dtype)
            )
            df.columns = self._header.data_cols_names
            df[self._header.timestamp_col_name] = time_vals
            return df
        else:
            raise NotImplementedError(
                f"Objects of type {type} are not supported"
            )

    @cached_property
    def body_decoder(self) -> msgspec.json.Decoder:
        return msgspec.json.Decoder(
            list[eds.FullAdmissionData], dec_hook=self._body_dec_hook
        )

    def decode(self) -> list[eds.FullAdmissionData]:
        return self.body_decoder.decode(self._body_bytes)


def import_admission_data_list(path: Path) -> list[eds.FullAdmissionData]:
    data_reader = AdmissionDataListReader.from_file(
        path=path, delimiter=ADMISSION_DATA_JSON_DELIMITER
    )
    return data_reader.decode()


class JsonReadyDataWriter:
    @cached_property
    def encoder(self) -> msgspec.json.Encoder:
        return msgspec.json.Encoder()

    def encode(self, obj: Any) -> bytes:
        return self.encoder.encode(obj)

    def export(self, obj: Any, path: Path):
        encoded_output = self.encode(obj)
        with path.open(mode="wb") as out_file:
            out_file.write(encoded_output)


def export_json_ready_object(obj: Any, path: Path):
    JsonReadyDataWriter().export(obj=obj, path=path)


EncodeType = TypeVar("EncodeType", bound=msgspec.Struct)


class StandardStructWriter(ABC):
    def __init__(self, struct_type: Callable[..., EncodeType]):
        self._struct_type = struct_type

    @staticmethod
    @abstractmethod
    def enc_hook(obj: Any) -> Any:
        pass

    @cached_property
    def encoder(self) -> msgspec.json.Encoder:
        return msgspec.json.Encoder(enc_hook=self.enc_hook)

    def encode(self, obj: EncodeType) -> bytes:
        return self.encoder.encode(obj)

    def export(self, obj: EncodeType, path: Path):
        encoded_data = self.encode(obj)
        with path.open(mode="wb") as out_file:
            out_file.write(encoded_data)


class FeatureArraysWriter(StandardStructWriter):
    def __init__(self):
        super().__init__(struct_type=eds.FeatureArrays)

    def enc_hook(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            raise NotImplementedError(
                f"Encoder does not support objects of type {type(obj)}"
            )


class ClassLabelsWriter(StandardStructWriter):
    def __init__(self):
        super().__init__(struct_type=eds.ClassLabels)

    @staticmethod
    def enc_hook(obj: Any) -> Any:
        pass  # ClassLabels object is json-ready


class MeasurementColumnNamesWriter(StandardStructWriter):
    def __init__(self):
        super().__init__(struct_type=eds.MeasurementColumnNames)

    @staticmethod
    def enc_hook(obj: Any) -> Any:
        pass  # MeasurementColumnNames object is json-ready


class PreprocessModuleSummaryWriter(StandardStructWriter):
    def __init__(self):
        super().__init__(struct_type=eds.PreprocessModuleSummary)

    @staticmethod
    def enc_hook(obj: Any) -> Any:
        pass  # PreprocessModuleSummary is json-ready


class TrainingCheckpointWriter(StandardStructWriter):
    def __init__(self):
        super().__init__(struct_type=mds.TrainingCheckpoint)

    @staticmethod
    def enc_hook(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, np.float64):
            return float(obj)


DecodeType = TypeVar("DecodeType", bound=msgspec.Struct)


class StandardStructReader(ABC):
    def __init__(self, struct_type: Callable[..., DecodeType]):
        self._struct_type = struct_type

    @staticmethod
    @abstractmethod
    def dec_hook(decode_type: Type, obj: Any) -> Any:
        pass

    @cached_property
    def decoder(self) -> msgspec.json.Decoder:
        return msgspec.json.Decoder(self._struct_type, dec_hook=self.dec_hook)

    def decode(self, obj: bytes) -> DecodeType:
        return self.decoder.decode(obj)

    def import_struct(self, path: Path) -> DecodeType:
        with path.open(mode="rb") as in_file:
            encoded_struct = in_file.read()
        return self.decode(obj=encoded_struct)


class FeatureArraysReader(StandardStructReader):
    def __init__(self):
        super().__init__(struct_type=eds.FeatureArrays)

    @staticmethod
    def dec_hook(decode_type: Type, obj: Any) -> Any:
        if decode_type is np.ndarray:
            return np.array(obj)
        else:
            raise NotImplementedError(
                f"Objects of type {type} are not supported"
            )


class ClassLabelsReader(StandardStructReader):
    def __init__(self):
        super().__init__(struct_type=eds.ClassLabels)

    @staticmethod
    def dec_hook(decode_type: Type, obj: Any) -> Any:
        # ClassLabels are json ready, so dec_hook should never be called
        raise NotImplementedError(f"Objects of type {type} are not supported")


class MeasurementColumnNamesReader(StandardStructReader):
    def __init__(self):
        super().__init__(struct_type=eds.MeasurementColumnNames)

    @staticmethod
    def dec_hook(decode_type: Type, obj: Any) -> Any:
        if decode_type is tuple[str]:
            return tuple(obj)
        else:
            raise NotImplementedError(
                f"Objects of type {type} are not supported"
            )


class TrainingCheckpointStorageReader(StandardStructReader):
    def __init__(self):
        super().__init__(struct_type=mds.TrainingCheckpointStorage)

    @staticmethod
    def dec_hook(decode_type: Type, obj: Any) -> Any:
        if decode_type is torch.Tensor:
            return torch.tensor(obj)
        else:
            raise NotImplementedError(
                f"Objects of type {type} are not supported"
            )
