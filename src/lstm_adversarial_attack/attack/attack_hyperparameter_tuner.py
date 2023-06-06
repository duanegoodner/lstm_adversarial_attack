import optuna
import torch
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AttackTuningRanges:
    kappa: tuple[float, float]
    lambda_1: tuple[float, float]
    optimizer_name: tuple[str, ...]
    learning_rate: tuple[float, float]


@dataclass
class AttackHyperParameterSettings:
    kappa: float
    lambda_1: float
    optimizer_name: str
    learning_rate: float

    @classmethod
    def from_optuna_active_trial(
        cls,
        trial: optuna.Trial,
        tuning_ranges: AttackTuningRanges
    ):
        return cls(
            kappa=trial.suggest_float(
                "kappa", *tuning_ranges.kappa, log=False
            ),
            lambda_1=trial.suggest_float(
                "lambda_1", *tuning_ranges.lambda_1, log=True),
            optimizer_name=trial.suggest_categorical(
                "optimizer_name", list(tuning_ranges.optimizer_name)
            ),
            learning_rate=trial.suggest_float(
                "learning_rate", *tuning_ranges.learning_rate, log=True
            )
        )



class AttackHyperParameterTuner:
    def __init__(
        self,
        device: torch.device,
        model_path: Path,
        checkpoint_path: Path,
        batch_size


    ):