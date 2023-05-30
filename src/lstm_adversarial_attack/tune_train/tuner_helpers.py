import sys
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from typing import Iterable, TypeVar

sys.path.append(str(Path(__file__).parent.parent.parent))
from lstm_adversarial_attack.data_structures import (
    EvalLog,
    OptimizeDirection,
)
from lstm_adversarial_attack.tune_train.standard_model_trainer import StandardModelTrainer


@dataclass
class TrainEvalDatasetPair:
    train: Dataset
    validation: Dataset


@dataclass
class X19MLSTMTuningRanges:
    log_lstm_hidden_size: tuple[int, int]
    lstm_act_options: tuple[str, ...]
    dropout: tuple[float, float]
    log_fc_hidden_size: tuple[int, int]
    fc_act_options: tuple[str, ...]
    optimizer_options: tuple[str, ...]
    learning_rate: tuple[float, float]
    log_batch_size: tuple[int, int]


@dataclass
class X19LSTMHyperParameterSettings:
    log_lstm_hidden_size: int
    lstm_act_name: str
    dropout: float
    log_fc_hidden_size: int
    fc_act_name: str
    optimizer_name: str
    learning_rate: float
    log_batch_size: int

    @classmethod
    def from_optuna(cls, trial, tuning_ranges: X19MLSTMTuningRanges):
        return cls(
            log_lstm_hidden_size=trial.suggest_int(
                "log_lstm_hidden_size",
                *tuning_ranges.log_lstm_hidden_size,
            ),
            lstm_act_name=trial.suggest_categorical(
                "lstm_act", list(tuning_ranges.lstm_act_options)
            ),
            dropout=trial.suggest_float("dropout", *tuning_ranges.dropout),
            log_fc_hidden_size=trial.suggest_int(
                "log_fc_hidden_size", *tuning_ranges.log_fc_hidden_size
            ),
            fc_act_name=trial.suggest_categorical(
                "fc_act", list(tuning_ranges.fc_act_options)
            ),
            optimizer_name=trial.suggest_categorical(
                "optimizer", list(tuning_ranges.optimizer_options)
            ),
            learning_rate=trial.suggest_float(
                "lr", *tuning_ranges.learning_rate, log=True
            ),
            log_batch_size=trial.suggest_int(
                "log_batch_size", *tuning_ranges.log_batch_size
            ),
        )


@dataclass
class ObjectiveFunctionTools:
    settings: X19LSTMHyperParameterSettings
    summary_writer: SummaryWriter
    cv_means_log: EvalLog
    trainers: list[StandardModelTrainer]


_T = TypeVar("_T")


class PerformanceSelector:
    _selection_dispatch = {
        OptimizeDirection.MIN: min,
        OptimizeDirection.MAX: max,
    }

    def __init__(self, optimize_direction: OptimizeDirection):
        self._optimize_direction = optimize_direction

    def choose_best_val(self, values: Iterable[_T]) -> _T:
        return self._selection_dispatch[self._optimize_direction](values)
