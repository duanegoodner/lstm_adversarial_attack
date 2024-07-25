from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable, Any

import msgspec
import optuna
import torch

from lstm_adversarial_attack.config import CONFIG_READER
import lstm_adversarial_attack.model.tuner_helpers as tuh


@dataclass
class AttackTuningRanges:
    kappa: tuple[float, float] = field(
        default_factory=lambda: tuple(
            CONFIG_READER.get_config_value("attack.tuning.kappa")
        )
    )
    lambda_1: tuple[float, float] = field(
        default_factory=lambda: tuple(
            CONFIG_READER.get_config_value("attack.tuning.lambda_1")
        )
    )
    optimizer_name: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            CONFIG_READER.get_config_value("attack.tuning.optimizer_options")
        )
    )

    learning_rate: tuple[float, float] = field(
        default_factory=lambda: tuple(
            CONFIG_READER.get_config_value("attack.tuning.learning_rate")
        )
    )
    log_batch_size: tuple[int, int] = field(
        default_factory=lambda: tuple(
            CONFIG_READER.get_config_value("attack.tuning.log_batch_size")
        )
    )


@dataclass
class AttackHyperParameterSettings:
    """
    Container for hyperparameters used by AdversarialAttackTrainer
    :param kappa: arameter from Equation 1 in Sun et al
        (https://arxiv.org/abs/1802.04822). Defines a margin by which alternate
        class logit value needs to exceed original class logit value in order
        to reduce loss function.
    :param lambda_1: L1 regularization constant applied to perturbations
    :param optimizer_name: name of optimizer (must match an attribute of
    torch.optim)
    :param learning_rate: learning rate to use during search for adversarial
    example_data
    :param log_batch_size: log (base 2) of batch size. Use log for easy
    Optuna param selection.

    """

    kappa: float
    lambda_1: float
    optimizer_name: str
    learning_rate: float
    log_batch_size: int

    @property
    def optimizer_constructor(self) -> Callable:
        return getattr(torch.optim, self.optimizer_name)


class BuildAttackHyperParameterSettings:
    @staticmethod
    def from_optuna_trial(
        trial: optuna.Trial, tuning_ranges: AttackTuningRanges
    ) -> AttackHyperParameterSettings:
        return AttackHyperParameterSettings(
            kappa=trial.suggest_float(
                "kappa", *tuning_ranges.kappa, log=False
            ),
            lambda_1=trial.suggest_float(
                "lambda_1", *tuning_ranges.lambda_1, log=True
            ),
            optimizer_name=trial.suggest_categorical(
                "optimizer_name", list(tuning_ranges.optimizer_name)
            ),
            learning_rate=trial.suggest_float(
                "learning_rate", *tuning_ranges.learning_rate, log=True
            ),
            log_batch_size=trial.suggest_int(
                "log_batch_size", *tuning_ranges.log_batch_size
            ),
        )


@dataclass
class AttackTunerDriverSettings:
    db_env_var_name: str
    num_trials: int
    epochs_per_batch: int
    max_num_samples: int
    sample_selection_seed: int
    pruner_name: str
    pruner_kwargs: dict[str, Any]
    sampler_name: str
    sampler_kwargs: dict[str, Any]
    objective_name: str
    max_perts: int  # used when objective_name = "max_num_nonzero_perts"

    @classmethod
    def from_config(cls, config_path: Path = None):
        # config_reader = ConfigReader(config_path=config_path)
        settings_fields = [
            field.name for field in fields(AttackTunerDriverSettings)
        ]
        constructor_kwargs = {
            field_name: CONFIG_READER.get_config_value(
                f"attack.tuner_driver_settings.{field_name}"
            )
            for field_name in settings_fields
        }
        return cls(**constructor_kwargs)


@dataclass
class AttackTunerDriverPaths:
    output_dir: Path

    @classmethod
    def from_config(cls):
        paths_fields = [field.name for field in fields(AttackTunerDriverPaths)]
        constructor_kwargs = {
            field_name: CONFIG_READER.read_path(
                f"attack.tuner_driver.{field_name}"
            )
            for field_name in paths_fields
        }
        return cls(**constructor_kwargs)


class AttackTunerDriverSummary(msgspec.Struct):
    preprocess_id: str
    attack_tuning_id: str
    cv_training_id: str
    model_hyperparameters: tuh.X19LSTMHyperParameterSettings
    settings: AttackTunerDriverSettings
    paths: AttackTunerDriverPaths
    study_name: str
    is_continuation: bool
    tuning_ranges: AttackTuningRanges
    model_training_result_dir: str
