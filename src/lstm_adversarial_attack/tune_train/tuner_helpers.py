import optuna
import sys
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from typing import Iterable, TypeVar

sys.path.append(str(Path(__file__).parent.parent.parent))
# import lstm_adversarial_attack.data_structures as ds
import lstm_adversarial_attack.config_settings as lcs
import lstm_adversarial_attack.tune_train.lstm_model_stc as lms
import lstm_adversarial_attack.tune_train.standard_model_trainer as smt
import lstm_adversarial_attack.data_structures as ds


@dataclass
class TrainEvalDatasetPair:
    """
    Container for a train dataset and eval dataset
    """
    train: Dataset
    validation: Dataset


@dataclass
class TrainEvalDataLoaderPair:
    """
    Container for a train dataloader and eval dataloader
    """
    train: DataLoader
    eval: DataLoader


@dataclass
class X19MLSTMTuningRanges:
    """
    Container of hyperparameter tuning ranges intended for use with Optuna
    """
    log_lstm_hidden_size: tuple[int, int] = lcs.TUNING_LOG_LSTM_HIDDEN_SIZE
    lstm_act_options: tuple[str, ...] = lcs.TUNING_LSTM_ACT_OPTIONS
    dropout: tuple[float, float] = lcs.TUNING_DROPOUT
    log_fc_hidden_size: tuple[int, int] = lcs.TUNING_LOG_FC_HIDDEN_SIZE
    fc_act_options: tuple[str, ...] = lcs.TUNING_FC_ACT_OPTIONS
    optimizer_options: tuple[str, ...] = lcs.TUNING_OPTIMIZER_OPTIONS
    learning_rate: tuple[float, float] = lcs.TUNING_LEARNING_RATE
    log_batch_size: tuple[int, int] = lcs.TUNING_LOG_BATCH_SIZE


@dataclass
class NonArchHyperParameterSettings:
    """
    Holds values of hyperparameters that are not part of model architecture
    """
    optimizer_name: str
    learning_rate: float
    log_batch_size: int


@dataclass
class X19LSTMHyperParameterSettings:
    """
    Hyperparameter values
    """
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
        cls, trial: optuna.Trial, tuning_ranges: X19MLSTMTuningRanges
    ):
        """
        Creates X19MLSTMTuningRanges object with getting Optuna suggestions
        :param trial: currently running optuna.Trial
        :param tuning_ranges: hyperparameter allowed ranges
        :return:
        """
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

    @property
    def non_arch_settings(self) -> NonArchHyperParameterSettings:
        """
        Returns the subset of settings not related to model architecture
        :return: values of non-architectural hyperparams
        """
        return NonArchHyperParameterSettings(
            optimizer_name=self.optimizer_name,
            learning_rate=self.learning_rate,
            log_batch_size=self.log_batch_size
        )


class X19LSTMBuilder:
    """
    Builds LSTM + 2 fully-connected layers model (as a torch.nn.Sequential)
    """
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
        """
        Builds model using hyperparameters assigned in constructor
        :return: a nn.Sequential model (takes VariableLengthFeatures as input)
        """
        return nn.Sequential(
            lms.BidirectionalX19LSTM(
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

    def build_for_model_graph(self) -> nn.Sequential:
        """
        Builds a model that does NOT take VariableLengthFeatures as input
        :return: nn.Sequential model
        """
        return nn.Sequential(
            lms.BidirectionalLSTMX19Graph(
                input_size=19,
                lstm_hidden_size=2 ** self.log_lstm_hidden_size,
            ),
            getattr(nn, self.lstm_act_name)(),
            nn.Dropout(p=self.dropout),
            nn.Linear(
                in_features=2 * (2 ** self.log_lstm_hidden_size),
                out_features=2 ** self.log_fc_hidden_size,
            ),
            getattr(nn, self.fc_act_name)(),
            nn.Linear(
                in_features=2 ** self.log_fc_hidden_size,
                out_features=2,
            ),
            nn.Softmax(dim=1),
        )


@dataclass
class ObjectiveFunctionTools:
    """
    Collection of tools needed by HyperparameterTuner.objective_fn
    """
    settings: X19LSTMHyperParameterSettings
    summary_writer: SummaryWriter
    cv_means_log: ds.EvalLog
    trainers: list[smt.StandardModelTrainer]


_T = TypeVar("_T")


class PerformanceSelector:
    """
    Generic selector of min or max vals. Used by HyperparameterTuner.
    """
    _selection_dispatch = {
        optuna.study.StudyDirection.MINIMIZE: min,
        "min": min,
        optuna.study.StudyDirection.MAXIMIZE: max,
        "max": max,
    }

    def __init__(self, optimize_direction: str):
        assert optimize_direction in self._selection_dispatch.keys()
        # assert (optimize_direction == "min") or (optimize_direction == "max")
        self._optimize_direction = optimize_direction

    def choose_best_val(self, values: Iterable[_T]) -> _T:
        return self._selection_dispatch[self._optimize_direction](values)
