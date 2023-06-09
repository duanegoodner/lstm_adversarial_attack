import sys

import optuna.trial
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from typing import Iterable, TypeVar

sys.path.append(str(Path(__file__).parent.parent.parent))
from lstm_aa.data_structures import (
    EvalLog,
    OptimizeDirection,
)
from lstm_aa.lstm_model_stc import BidirectionalX19LSTM
from lstm_aa.tune_train.standard_model_trainer import (
    StandardModelTrainer,
)


@dataclass
class TrainEvalDatasetPair:
    train: Dataset
    validation: Dataset


@dataclass
class TrainEvalDataLoaderPair:
    train: DataLoader
    eval: DataLoader


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

# TODO segregate into params for model and params for trainer
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
    def from_optuna_active_trial(
        cls, trial, tuning_ranges: X19MLSTMTuningRanges
    ):
        return cls(
            log_lstm_hidden_size=trial.suggest_int(
                "log_lstm_hidden_size",
                *tuning_ranges.log_lstm_hidden_size,
            ),
            lstm_act_name=trial.suggest_categorical(
                "lstm_act_name", list(tuning_ranges.lstm_act_options)
            ),
            dropout=trial.suggest_float("dropout", *tuning_ranges.dropout),
            log_fc_hidden_size=trial.suggest_int(
                "log_fc_hidden_size", *tuning_ranges.log_fc_hidden_size
            ),
            fc_act_name=trial.suggest_categorical(
                "fc_act_name", list(tuning_ranges.fc_act_options)
            ),
            optimizer_name=trial.suggest_categorical(
                "optimizer_name", list(tuning_ranges.optimizer_options)
            ),
            learning_rate=trial.suggest_float(
                "learning_rate", *tuning_ranges.learning_rate, log=True
            ),
            log_batch_size=trial.suggest_int(
                "log_batch_size", *tuning_ranges.log_batch_size
            ),
        )


class X19LSTMBuilder:
    def __init__(
        self,
        settings: X19LSTMHyperParameterSettings,
    ):
        self.log_lstm_hidden_size = settings.log_lstm_hidden_size
        self.lstm_act_name = settings.lstm_act_name
        self.dropout = settings.dropout
        self.log_fc_hidden_size = settings.log_fc_hidden_size
        self.fc_act_name = settings.fc_act_name

    def build(self) -> nn.Sequential:
        return nn.Sequential(
            BidirectionalX19LSTM(
                input_size=19,
                lstm_hidden_size=2**self.log_lstm_hidden_size,
            ),
            getattr(nn, self.lstm_act_name)(),
            nn.Dropout(p=self.dropout),
            nn.Linear(
                in_features=2 * (2**self.log_lstm_hidden_size),
                out_features=2**self.log_fc_hidden_size,
            ),
            getattr(nn, self.fc_act_name)(),
            nn.Linear(
                in_features=2**self.log_fc_hidden_size,
                out_features=2,
            ),
            nn.Softmax(dim=1),
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
        "min": min,
        "max": max,
    }

    def __init__(self, optimize_direction: str):
        assert (optimize_direction == "min") or (optimize_direction == "max")
        self._optimize_direction = optimize_direction

    def choose_best_val(self, values: Iterable[_T]) -> _T:
        return self._selection_dispatch[self._optimize_direction](values)
