import collections
import sys
from abc import ABC
from dataclasses import dataclass, fields
from enum import Enum, auto
from pathlib import Path
from typing import Any, Type

import msgspec
import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.model.tuner_helpers as tuh
import lstm_adversarial_attack.utils.msgspec_io as mio
from lstm_adversarial_attack.config.read_write import CONFIG_READER, PATH_CONFIG_READER


class MsgSpecStructWithDict(msgspec.Struct):

    @property
    def __dict__(self) -> dict:
        return {field: getattr(self, field) for field in self.__struct_fields__}


# @dataclass
# class VariableLengthFeatures:
#     features: torch.tensor
#     lengths: torch.tensor


@dataclass
class ClassificationScores:
    accuracy: float
    auc: float
    precision: float
    recall: float
    f1: float

    def __str__(self) -> str:
        return (
            f"Accuracy:\t{self.accuracy:.4f}\n"
            f"AUC:\t\t{self.auc:.4f}\n"
            f"Precision:\t{self.precision:.4f}\n"
            f"Recall:\t\t{self.recall:.4f}\n"
            f"F1:\t\t\t{self.f1:.4f}"
        )


class TrainEpochResult(msgspec.Struct):
    loss: float

    @classmethod
    def mean(cls, results: list["TrainEpochResult"]) -> "TrainEpochResult":
        return cls(loss=np.mean([item.loss for item in results]))


# @dataclass
class EvalEpochResult(MsgSpecStructWithDict):
    validation_loss: float
    accuracy: float
    auc: float
    precision: float
    recall: float
    f1: float

    @classmethod
    def from_loss_and_scores(
        cls, validation_loss: float, scores: ClassificationScores
    ):
        return cls(
            validation_loss=validation_loss,
            accuracy=scores.accuracy,
            auc=scores.auc,
            precision=scores.precision,
            recall=scores.recall,
            f1=scores.f1,
        )

    def __add__(self, other):
        if isinstance(other, EvalEpochResult):
            sum_dict = {
                key: self.__dict__[key] + other.__dict__[key]
                for key in self.__dict__.keys()
            }
            return EvalEpochResult(**sum_dict)
        else:
            raise TypeError("Unsupported operand type")

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            else:
                quotient_dict = {
                    key: self.__dict__[key] / other
                    for key in self.__dict__.keys()
                }
            return EvalEpochResult(**quotient_dict)
        else:
            raise TypeError("Unsupported operand type")

    @classmethod
    def mean(cls, results: list["EvalEpochResult"]) -> "EvalEpochResult":
        return cls(
            validation_loss=np.mean(
                [item.validation_loss for item in results]
            ),
            accuracy=np.mean([item.accuracy for item in results]),
            auc=np.mean([item.auc for item in results]),
            precision=np.mean([item.precision for item in results]),
            recall=np.mean([item.recall for item in results]),
            f1=np.mean([item.f1 for item in results]),
        )

    def __str__(self) -> str:
        return (
            f"Loss:\t\t{self.validation_loss:.4f}\n"
            f"Accuracy:\t{self.accuracy:.4f}\n"
            f"AUC:\t\t{self.auc:.4f}\n"
            f"Precision:\t{self.precision:.4f}\n"
            f"Recall:\t\t{self.recall:.4f}\n"
            f"F1:\t\t\t{self.f1:.4f}"
        )


class TrainLogEntry(msgspec.Struct):
    epoch: int
    result: TrainEpochResult


class EvalLogEntry(msgspec.Struct):
    epoch: int
    result: EvalEpochResult


@dataclass
class TrainEvalLog(ABC):
    data: list[TrainLogEntry] | list[EvalLogEntry] = None

    def __post_init__(self):
        if self.data is None:
            self.data = []

    def update(self, entry: TrainLogEntry | EvalLogEntry):
        self.data.append(entry)

    @property
    def latest_entry(self) -> TrainLogEntry | EvalLogEntry | None:
        if len(self.data) == 0:
            return None
        else:
            return self.data[-1]

    def get_all_entries_of_attribute(self, attr: str) -> list[float]:
        return [getattr(entry, name=attr) for entry in self.data]

    def data_attribute_mean(self, attr: str) -> float:
        return np.mean(self.get_all_entries_of_attribute(attr=attr))


@dataclass
class TrainLog(TrainEvalLog):
    data: list[TrainLogEntry] = None


@dataclass
class EvalLog(TrainEvalLog):
    data: list[EvalLogEntry] = None


@dataclass
class TrainEvalLogPair:
    train: TrainLog
    eval: EvalLog


@dataclass
class CVTrialLogs:
    # trial: optuna.Trial
    cv_means_log: EvalLog
    trainer_logs: list[TrainEvalLogPair]


class OptimizeDirection(Enum):
    MIN = auto()
    MAX = auto()


@dataclass
class FullEvalResult:
    metrics: EvalEpochResult
    y_pred: torch.tensor
    y_score: torch.tensor
    y_true: torch.tensor


class OptimizerStateDict(msgspec.Struct):
    param_groups: list[dict[str, Any]]
    state: dict[int, dict[str, torch.Tensor]]


class ModelStateDictInfo(msgspec.Struct):
    unordered_dict: dict[str, torch.Tensor]
    key_order: list

    @classmethod
    def from_possibly_ordered_state_dict(
        cls, ordered_state_dict: collections.OrderedDict | dict
    ):
        return cls(
            unordered_dict=dict(ordered_state_dict),
            key_order=list(ordered_state_dict.keys()),
        )

    @property
    def ordered_dict(self) -> collections.OrderedDict:
        return collections.OrderedDict(
            (key, self.unordered_dict[key]) for key in self.key_order
        )


class TrainingCheckpointStorage(msgspec.Struct):
    epoch_num: int
    train_log_entry: TrainLogEntry
    eval_log_entry: EvalLogEntry
    state_dict_info: ModelStateDictInfo
    optimizer_state_dict: OptimizerStateDict

    @property
    def state_dict(self) -> collections.OrderedDict:
        return self.state_dict_info.ordered_dict


class TrainingCheckpointStorageReader(mio.StandardStructReader):
    def __init__(self):
        super().__init__(struct_type=TrainingCheckpointStorage)

    @staticmethod
    def dec_hook(decode_type: Type, obj: Any) -> Any:
        if decode_type is torch.Tensor:
            return torch.tensor(obj)
        else:
            raise NotImplementedError(
                f"Objects of type {decode_type} are not supported"
            )


class TrainingCheckpoint(msgspec.Struct):
    epoch_num: int
    train_log_entry: TrainLogEntry
    eval_log_entry: EvalLogEntry
    state_dict: collections.OrderedDict
    optimizer_state_dict: OptimizerStateDict

    def to_storage(self) -> TrainingCheckpointStorage:
        return TrainingCheckpointStorage(
            epoch_num=self.epoch_num,
            train_log_entry=self.train_log_entry,
            eval_log_entry=self.eval_log_entry,
            state_dict_info=ModelStateDictInfo.from_possibly_ordered_state_dict(
                ordered_state_dict=self.state_dict
            ),
            optimizer_state_dict=self.optimizer_state_dict,
        )

    @classmethod
    def from_storage(
        cls, training_checkpoint_storage: TrainingCheckpointStorage
    ):
        return cls(
            epoch_num=training_checkpoint_storage.epoch_num,
            train_log_entry=training_checkpoint_storage.train_log_entry,
            eval_log_entry=training_checkpoint_storage.eval_log_entry,
            state_dict=training_checkpoint_storage.state_dict,
            optimizer_state_dict=training_checkpoint_storage.optimizer_state_dict,
        )


class TrainingCheckpointWriter(mio.StandardStructWriter):
    def __init__(self):
        super().__init__(struct_type=TrainingCheckpoint)

    @staticmethod
    def enc_hook(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, np.float64):
            return float(obj)





@dataclass
class CrossValidatorDriverSettings:
    epochs_per_fold: int
    num_folds: int
    eval_interval: int
    kfold_random_seed: int
    fold_class_name: str
    collate_fn_name: str
    single_fold_eval_fraction: float

    @classmethod
    def from_config(cls, config_path: Path = None):
        settings_fields = [
            item.name for item in fields(CrossValidatorDriverSettings)
        ]
        constructor_kwargs = {
            field_name: CONFIG_READER.get_value(
                f"model.cv_driver_settings.{field_name}"
            )
            for field_name in settings_fields
        }
        return cls(**constructor_kwargs)


@dataclass
class CrossValidatorDriverPaths:
    output_dir: str

    @classmethod
    def from_config(cls, config_path: Path = None):
        paths_fields = [
            item.name for item in fields(CrossValidatorDriverPaths)
        ]
        constructor_kwargs = {
            field_name: PATH_CONFIG_READER.read_path(
                f"model.cv_driver.{field_name}"
            )
            for field_name in paths_fields
        }
        return cls(**constructor_kwargs)


@dataclass
class ModelTunerDriverSettings:
    num_trials: int
    num_folds: int
    num_cv_epochs: int
    epochs_per_fold: int
    kfold_random_seed: int
    cv_mean_tensorboard_metrics: list[str]
    performance_metric: str
    optimization_direction_label: str
    pruner_name: str
    pruner_kwargs: dict[str, Any]
    sampler_name: str
    sampler_kwargs: dict[str, Any]
    db_env_var_name: str
    fold_class_name: str
    collate_fn_name: str
    tuning_ranges: dict[str, Any]

    @classmethod
    def from_config(cls):
        settings_fields = [
            item.name for item in fields(ModelTunerDriverSettings)
        ]
        constructor_kwargs = {
            field_name: CONFIG_READER.get_value(
                f"model.tuner_driver.{field_name}"
            )
            for field_name in settings_fields
        }
        return cls(**constructor_kwargs)


@dataclass
class ModelTunerDriverPaths:
    output_dir: str

    @classmethod
    def from_config(cls):
        paths_fields = [item.name for item in fields(ModelTunerDriverPaths)]
        constructor_kwargs = {
            field_name: PATH_CONFIG_READER.read_path(
                f"model.tuner_driver.{field_name}"
            )
            for field_name in paths_fields
        }
        return cls(**constructor_kwargs)


class TunerDriverSummary(msgspec.Struct):
    preprocess_id: str
    model_tuning_id: str
    settings: ModelTunerDriverSettings
    paths: ModelTunerDriverPaths
    study_name: str
    is_continuation: bool
    device_name: str


class TunerDriverSummaryIO(mio.MsgspecIO):
    def __init__(self):
        super().__init__(msgspec_struct_type=TunerDriverSummary)

    @staticmethod
    def enc_hook(obj: Any) -> Any:
        if isinstance(obj, tuh.X19MLSTMTuningRanges):
            return obj.__dict__
        else:
            raise NotImplementedError(
                f"Encoder does not support objects of type {type(obj)}"
            )

    @staticmethod
    def dec_hook(decode_type: Type, obj: Any) -> Any:
        if decode_type is tuple[str]:
            return tuple(obj)
        if decode_type is tuh.X19MLSTMTuningRanges:
            return tuh.X19MLSTMTuningRanges(**obj)
        else:
            raise NotImplementedError(
                f"Decoder does not support objects of type {type(obj)}"
            )


TUNER_DRIVER_SUMMARY_IO = TunerDriverSummaryIO()


class CrossValidatorDriverSummary(msgspec.Struct):
    preprocess_id: str
    model_tuning_id: str
    model_tuning_trial_number: int
    cv_training_id: str
    model_hyperparameters: tuh.X19LSTMHyperParameterSettings
    settings: CrossValidatorDriverSettings
    paths: CrossValidatorDriverPaths


class CrossValidatorDriverSummaryIO(mio.MsgspecIO):
    def __init__(self):
        super().__init__(msgspec_struct_type=CrossValidatorDriverSummary)

    @staticmethod
    def enc_hook(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, np.float64):
            return float(obj)
        else:
            raise NotImplementedError(
                f"Encoder does not support objects of type {type(obj)}"
            )

    @staticmethod
    def dec_hook(decode_type: Type, obj: Any) -> Any:
        if decode_type is tuple[str]:
            return tuple(obj)
        if decode_type is Path:
            return Path(obj)


CROSS_VALIDATOR_DRIVER_SUMMARY_IO = CrossValidatorDriverSummaryIO()

