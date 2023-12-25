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
    output_dir: str

    def to_dict(self):
        return {
            "settings": self.settings,
            "paths": self.paths,
            "study_name": self.study_name,
            "is_continuation": self.is_continuation,
            "device_name": self.device_name,
            "output_dir": self.output_dir,
        }


class AttackTunerDriverSummary(msgspec.Struct):
    hyperparameters_path: str
    objective_name: str
    objective_extra_kwargs: dict[str, Any]
    db_env_var_name: str
    study_name: str
    is_continuation: bool
    tuning_ranges: ads.AttackTuningRanges
    epochs_per_batch: int
    max_num_samples: int
    sample_selection_seed: int
    training_result_dir: str
    pruner_name: str
    pruner_kwargs: dict[str, Any]
    sampler_name: str
    sampler_kwargs: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hyperparameters_path": self.hyperparameters_path,
            "objective_name": self.objective_name,
            "objective_extra_kwargs": self.objective_extra_kwargs,
            "db_env_var_name": self.db_env_var_name,
            "study_name": self.study_name,
            "is_continuation": self.is_continuation,
            "tuning_ranges": self.tuning_ranges,
            "epochs_per_batch": self.epochs_per_batch,
            "max_num_samples": self.max_num_samples,
            "sample_selection_seed": self.sample_selection_seed,
            "training_result_dir": self.training_result_dir,
            "pruner_name": self.pruner_name,
            "pruner_kwargs": self.pruner_kwargs,
            "sampler_name": self.sampler_name,
            "sampler_kwargs": self.sampler_kwargs
        }



