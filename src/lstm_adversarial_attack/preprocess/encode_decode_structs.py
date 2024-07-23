from typing import Any

import msgspec
import numpy as np
import pandas as pd

import lstm_adversarial_attack.attack.attack_data_structs as ads
import lstm_adversarial_attack.model.model_data_structs as mds
import lstm_adversarial_attack.model.tuner_helpers as tuh


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
    preprocess_id: str
    settings: dict[str, Any]
    paths: dict[str, str]
    study_name: str
    is_continuation: bool
    device_name: str

    def to_dict(self):
        return {
            "preprocess_id": self.preprocess_id,
            "settings": self.settings,
            "paths": self.paths,
            "study_name": self.study_name,
            "is_continuation": self.is_continuation,
            "device_name": self.device_name,
        }


class CrossValidatorDriverSummary(msgspec.Struct):
    preprocess_id: str
    tuning_study_name: str
    cv_driver_id: str
    model_hyperparameters: tuh.X19LSTMHyperParameterSettings
    settings: mds.CrossValidatorDriverSettings
    paths: mds.CrossValidatorDriverPaths

    def to_dict(self):
        return {
            "preprocess_id": self.preprocess_id,
            "tuning_study_name": self.tuning_study_name,
            "cv_driver_id": self.cv_driver_id,
            "model_hyperparameters": self.model_hyperparameters,
            "settings": self.settings,
            "paths": self.paths,
        }


class AttackTunerDriverSummary(msgspec.Struct):
    settings: ads.AttackTunerDriverSettings
    paths: ads.AttackTunerDriverPaths
    preprocess_id: str
    study_name: str
    is_continuation: bool
    tuning_ranges: ads.AttackTuningRanges
    model_training_result_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "settings": self.settings,
            "paths": self.paths,
            "preprocess_id": self.preprocess_id,
            "study_name": self.study_name,
            "is_continuation": self.is_continuation,
            "tuning_ranges": self.tuning_ranges,
            "model_training_result_dir": self.model_training_result_dir,
        }



