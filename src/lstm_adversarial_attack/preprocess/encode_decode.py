import time
from functools import cached_property
from pathlib import Path
from typing import Any, Type

import msgspec
import numpy as np
import pandas as pd

import lstm_adversarial_attack.config_paths as cfp
import lstm_adversarial_attack.config_settings as cfs
import lstm_adversarial_attack.preprocess.encode_decode_structs as eds

# TODO Consider creating DataWriter base class with encoder abstractmethod and
#  encode() & export() concrete methods


class AdmissionDataWriter:
    def __init__(self, delimiter: str = cfs.ADMISSION_DATA_JSON_DELIMITER):
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
        self, full_admission_data_list: list[eds.NewFullAdmissionData]
    ) -> bytes:
        example_df = full_admission_data_list[0].time_series
        timestamp_col_name = "charttime"
        timestamp_dtype = example_df[timestamp_col_name].dtype.name
        data_cols_names = [
            item for item in example_df.columns if item != timestamp_col_name
        ]
        data_only_df = example_df[data_cols_names]
        data_cols_dtype = np.unique(data_only_df.dtypes).item().name

        header = eds.NewFullAdmissionDataListHeader(
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

    def export(
        self,
        full_admission_data_list: list[eds.NewFullAdmissionData],
        path: Path,
    ):
        encoded_header_and_body = self.encode(full_admission_data_list)
        with path.open(mode="wb") as out_file:
            out_file.write(encoded_header_and_body)


class AdmissionDataListReader:
    def __init__(
        self,
        encoded_data: bytes,
        delimiter: str = cfs.ADMISSION_DATA_JSON_DELIMITER,
    ):
        self._encoded_data = encoded_data
        self._delimiter = delimiter

    @classmethod
    def from_file(
        cls, path: Path, delimiter: str = cfs.ADMISSION_DATA_JSON_DELIMITER
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
        return msgspec.json.Decoder(eds.NewFullAdmissionDataListHeader)

    @cached_property
    def _header(self) -> eds.NewFullAdmissionDataListHeader:
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
            list[eds.NewFullAdmissionData], dec_hook=self._body_dec_hook
        )

    def decode(self) -> list[eds.NewFullAdmissionData]:
        return self.body_decoder.decode(self._body_bytes)


ADMISSION_DATA_WRITER = AdmissionDataWriter()


def export_admission_data_list(
    data_obj: list[eds.NewFullAdmissionData], path: Path
):
    ADMISSION_DATA_WRITER.export(full_admission_data_list=data_obj, path=path)


def import_admission_data_list(path: Path) -> list[eds.NewFullAdmissionData]:
    data_reader = AdmissionDataListReader.from_file(
        path=path, delimiter=cfs.ADMISSION_DATA_JSON_DELIMITER
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


JSON_READY_DATA_WRITER = JsonReadyDataWriter()


def export_json_ready_object(obj: Any, path: Path):
    JSON_READY_DATA_WRITER.export(obj=obj, path=path)


class FeatureArraysDataWriter:
    def enc_hook(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            raise NotImplementedError(
                f"Encoder does not support objects of type {type(obj)}"
            )

    @cached_property
    def encoder(self) -> msgspec.json.Encoder:
        return msgspec.json.Encoder(enc_hook=self.enc_hook)

    def encode(self, obj: Any) -> bytes:
        return self.encoder.encode(obj)

    def export(self, obj: Any, path: Path):
        encoded_data = self.encode(obj)
        with path.open(mode="wb") as out_file:
            out_file.write(encoded_data)


FEATURE_ARRAYS_DATA_WRITER = FeatureArraysDataWriter()


def export_feature_arrays(np_arrays: list[np.ndarray], path: Path):
    FEATURE_ARRAYS_DATA_WRITER.export(obj=np_arrays, path=path)


class FeatureArraysDataReader:
    @staticmethod
    def dec_hook(type: Type, obj: Any) -> Any:
        if type is np.ndarray:
            return np.array(obj)
        else:
            raise NotImplementedError(
                f"Objects of type {type} are not supported")

    @cached_property
    def decoder(self) -> msgspec.json.Decoder:
        return msgspec.json.Decoder(eds.FeatureArrays, dec_hook=self.dec_hook)

    def decode(self, obj: bytes) -> eds.FeatureArrays:
        return self.decoder.decode(obj)

    def import_object(self, path: Path) -> eds.FeatureArrays:
        with path.open(mode="rb") as in_file:
            encoded_object = in_file.read()
        return self.decode(encoded_object)




if __name__ == "__main__":
    import_path = cfp.FULL_ADMISSION_LIST_OUTPUT / "full_admission_list.json"
    json_import_start = time.time()
    result = import_admission_data_list(path=import_path)
    json_import_end = time.time()
