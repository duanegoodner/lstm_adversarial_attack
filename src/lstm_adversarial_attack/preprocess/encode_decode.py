import collections
import time
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Any, Type, TypeVar, Callable

import msgspec
import numpy as np
import pandas as pd
import torch

import lstm_adversarial_attack.config_paths as cfp
import lstm_adversarial_attack.config_settings as cfs
import lstm_adversarial_attack.preprocess.encode_decode_structs as eds
import lstm_adversarial_attack.data_structures as ds
import lstm_adversarial_attack.tune_train.tuner_helpers as tuh

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

    def export(self, obj: Any, path: Path):
        encoded_data = self.encode(obj)
        with path.open(mode="wb") as out_file:
            out_file.write(encoded_data)


def export_admission_data_list(
    admission_data_list: list[eds.NewFullAdmissionData], path: Path
):
    AdmissionDataWriter().export(obj=admission_data_list, path=path)


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


class TunerDriverSummaryWriter(StandardStructWriter):
    def __init__(self):
        super().__init__(struct_type=eds.TunerDriverSummary)

    @staticmethod
    def enc_hook(obj: Any) -> Any:
        if isinstance(obj, tuh.X19MLSTMTuningRanges):
            return obj.__dict__
        else:
            raise NotImplementedError(
                f"Encoder does not support objects of type {type(obj)}"
            )


class TrainingCheckpointWriter(StandardStructWriter):
    def __init__(self):
        super().__init__(struct_type=ds.TrainingCheckpoint)

    @staticmethod
    def enc_hook(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, np.float64):
            return float(obj)


class X19LSTMHyperParameterSettingsWriter(StandardStructWriter):
    def __init__(self):
        super().__init__(struct_type=tuh.X19LSTMHyperParameterSettings)

    @staticmethod
    def enc_hook(obj: Any) -> Any:
        pass  #json ready


class AttackTunerDriverSummaryWriter(StandardStructWriter):
    def __init__(self):
        super().__init__(struct_type=eds.AttackTunerDriverSummary)

    @staticmethod
    def enc_hook(obj: Any) -> Any:
        pass #json ready


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


class TunerDriverSummaryReader(StandardStructReader):
    def __init__(self):
        super().__init__(struct_type=eds.TunerDriverSummary)

    @staticmethod
    def dec_hook(decode_type: Type, obj: Any) -> Any:
        if decode_type is tuple[str]:
            return tuple(obj)
        if decode_type is tuh.X19MLSTMTuningRanges:
            return tuh.X19MLSTMTuningRanges(**obj)
        else:
            raise NotImplementedError(
                f"Objects of type {type} are not supported"
            )


class AttackTunerDriverSummaryReader(StandardStructReader):
    def __init__(self):
        super().__init__(struct_type=eds.AttackTunerDriverSummary)

    @staticmethod
    def dec_hook(decode_type: Type, obj: Any) -> Any:
        if decode_type is tuple:
            return tuple(obj)
        if decode_type is ads.AttackTuningRanges:
            return ads.AttackTuningRanges(**obj)


class TrainingCheckpointStorageReader(StandardStructReader):
    def __init__(self):
        super().__init__(struct_type=ds.TrainingCheckpointStorage)

    @staticmethod
    def dec_hook(decode_type: Type, obj: Any) -> Any:
        if decode_type is torch.Tensor:
            return torch.tensor(obj)
        else:
            raise NotImplementedError(
                f"Objects of type {type} are not supported"
            )


class X19LSTMHyperParameterSettingsReader(StandardStructReader):
    def __init__(self):
        super().__init__(struct_type=tuh.X19LSTMHyperParameterSettings)

    @staticmethod
    def dec_hook(decode_type: Type, obj: Any) -> Any:
        # all expected types are supported
        raise NotImplementedError(
            f"Objects of type {type} are not supported"
        )


if __name__ == "__main__":
    checkpoint_from_pickle = torch.load(
        cfp.CV_ASSESSMENT_OUTPUT_DIR
        / "cv_training_20230901215330316505"
        / "checkpoints"
        / "fold_2"
        / "2023-09-01_21_55_35.278089.tar"
    )

    json_output_path = (
        cfp.CV_ASSESSMENT_OUTPUT_DIR
        / "cv_training_20230901215330316505"
        / "checkpoints"
        / "fold_2"
        / "training_checkpoint_20230901215535284037.json"
    )

    checkpoint_storage_from_json = (
        TrainingCheckpointStorageReader().import_struct(path=json_output_path)
    )

    checkpoint_from_json = ds.TrainingCheckpoint.from_storage(
        training_checkpoint_storage=checkpoint_storage_from_json
    )

    for key, val in checkpoint_from_json.state_dict.items():
        checkpoint_from_json.state_dict[key] = val.to("cuda:0")
        print(
            torch.equal(
                checkpoint_from_json.state_dict[key],
                checkpoint_from_pickle["state_dict"][key],
            )
        )

    # import_path = cfp.FULL_ADMISSION_LIST_OUTPUT / "full_admission_list.json"
    # json_import_start = time.time()
    # result = import_admission_data_list(path=import_path)
    # json_import_end = time.time()

    # my_features = [
    #     np.array([[1.1, 2.2, 3.3]]),
    #     np.array([[4.4, 5.5, 6.6], [7.7, 8.8, 9.9]]),
    # ]
    #
    # feature_arrays = eds.FeatureArrays(data=my_features)
    # encoded_feature_arrays = FeatureArraysWriter().encode(feature_arrays)
    # decoded_features = FeatureArraysReader().decode(encoded_feature_arrays)
    #
    # my_class_labels = [0, 0, 1]
    # class_labels = eds.ClassLabels(data=my_class_labels)
    # encoded_class_labels = ClassLabelsWriter().encode(class_labels)
    # decoded_class_labels = ClassLabelsReader().decode(encoded_class_labels)
    #
    # my_col_names = ("col_one", "col_two", "col_three")
    # col_names = eds.MeasurementColumnNames(data=my_col_names)
    # encoded_col_names = MeasurementColumnNamesWriter().encode(col_names)
    # decoded_col_names = MeasurementColumnNamesReader().decode(
    #     encoded_col_names
    # )

    # tuner_driver_summary = TunerDriverSummaryReader().import_struct(
    #     path=cfp.HYPERPARAMETER_OUTPUT_DIR
    #     / "2023-08-17_15_05_51.518571"
    #     / "tuner_driver_summary_2023-08-17_15_05_51.519082.json"
    # )
