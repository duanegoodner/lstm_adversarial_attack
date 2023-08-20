import msgspec
import numpy as np
import torch
from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterable, TypeVar, Type, Callable, Any


@dataclass
class VariableLengthFeatures:
    features: torch.tensor
    lengths: torch.tensor


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


@dataclass
class TrainEpochResult:
    loss: float

    @classmethod
    def mean(cls, results: list["TrainEpochResult"]) -> "TrainEpochResult":
        return cls(loss=np.mean([item.loss for item in results]))


@dataclass
class EvalEpochResult:
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


@dataclass
class TrainLogEntry:
    epoch: int
    result: TrainEpochResult


@dataclass
class EvalLogEntry:
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

