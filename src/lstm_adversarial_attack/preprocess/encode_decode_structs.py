from typing import Any

import msgspec
import numpy as np
import pandas as pd

import lstm_adversarial_attack.attack.attack_data_structs as ads


class FullAdmissionData(msgspec.Struct):
    """
    Container used as elements list build by FullAdmissionListBuilder
    """
    subject_id: int
    hadm_id: int
    icustay_id: int
    admittime: pd.Timestamp
    dischtime: pd.Timestamp
    hospital_expire_flag: int
    intime: pd.Timestamp
    outtime: pd.Timestamp
    time_series: pd.DataFrame


class DecomposedTimeSeries(msgspec.Struct):
    index: list[int]
    time_vals: list[pd.Timestamp]
    data: list[list[float]]


class FullAdmissionDataListHeader(msgspec.Struct):
    timestamp_col_name: str
    timestamp_dtype: str
    data_cols_names: list[str]
    data_cols_dtype: str


class FeatureArrays(msgspec.Struct):
    data: list[np.ndarray]


class ClassLabels(msgspec.Struct):
    data: list[int] = None


class MeasurementColumnNames(msgspec.Struct):
    data: tuple[str, ...]


class PreprocessModuleSummary(msgspec.Struct):
    output_dir: str
    output_constructors: dict[str, str]
    resources: dict[str, dict[str, str]]
    settings: dict[str, Any]


class TunerDriverSummary(msgspec.Struct):
    settings: dict[str, Any]
    paths: dict[str, str]
    study_name: str
    is_continuation: bool
    device_name: str

    def to_dict(self):
        return {
            "settings": self.settings,
            "paths": self.paths,
            "study_name": self.study_name,
            "is_continuation": self.is_continuation,
            "device_name": self.device_name,
        }


class AttackTunerDriverSummary(msgspec.Struct):
    settings: ads.AttackTunerDriverSettings
    paths: ads.AttackTunerDriverPaths
    study_name: str
    is_continuation: bool
    tuning_ranges: ads.AttackTuningRanges
    model_training_result_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "settings": self.settings,
            "paths": self.paths,
            "study_name": self.study_name,
            "is_continuation": self.is_continuation,
            "tuning_ranges": self.tuning_ranges,
            "model_training_result_dir": self.model_training_result_dir,
        }



