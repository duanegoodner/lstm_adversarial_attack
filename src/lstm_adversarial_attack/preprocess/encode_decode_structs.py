from typing import Any

import msgspec
import numpy as np
import pandas as pd

import lstm_adversarial_attack.tune_train.tuner_helpers as tuh


class NewFullAdmissionData(msgspec.Struct):
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


class NewFullAdmissionDataListHeader(msgspec.Struct):
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
    collate_fn_name: str
    is_continuation: bool
    cv_mean_metrics_of_interest: tuple[str, ...]
    device_name: str
    epochs_per_fold: int
    fold_class_name: str
    output_dir: str
    sampler_name: str
    sampler_kwargs: dict[str, Any]
    kfold_random_seed: int
    num_cv_epochs: int
    num_folds: int
    optimization_direction_label: str
    performance_metric: str
    pruner_name: str
    tuning_ranges: tuh.X19MLSTMTuningRanges

    def to_dict(self):
        return {
            "collate_fn_name": self.collate_fn_name,
            "is_continuation": self.is_continuation,
            "cv_mean_metrics_of_interest": self.cv_mean_metrics_of_interest,
            "device_name": self.device_name,
            "epochs_per_fold": self.epochs_per_fold,
            "fold_class_name": self.fold_class_name,
            "output_dir": self.output_dir,
            "sampler_name": self.sampler_name,
            "sampler_kwargs": self.sampler_kwargs,
            "kfold_random_seed": self.kfold_random_seed,
            "num_cv_epochs": self.num_cv_epochs,
            "num_folds": self.num_folds,
            "optimization_direction_label": self.optimization_direction_label,
            "performance_metric": self.performance_metric,
            "pruner_name": self.pruner_name,
            "tuning_ranges": self.tuning_ranges
        }


