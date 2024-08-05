from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Type

import msgspec
import numpy as np
import optuna
import torch

import lstm_adversarial_attack.model.model_data_structs as mds
import lstm_adversarial_attack.model.tuner_helpers as tuh
import lstm_adversarial_attack.utils.msgspec_io as mio
from lstm_adversarial_attack.config import CONFIG_READER, PATH_CONFIG_READER


@dataclass
class AttackTuningRanges:
    kappa: tuple[float, float] = field(
        default_factory=lambda: tuple(
            CONFIG_READER.get_config_value("attack.tuning.ranges.kappa")
        )
    )
    lambda_1: tuple[float, float] = field(
        default_factory=lambda: tuple(
            CONFIG_READER.get_config_value("attack.tuning.ranges.lambda_1")
        )
    )
    optimizer_name: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            CONFIG_READER.get_config_value("attack.tuning.ranges.optimizer_options")
        )
    )

    learning_rate: tuple[float, float] = field(
        default_factory=lambda: tuple(
            CONFIG_READER.get_config_value("attack.tuning.ranges.learning_rate")
        )
    )
    log_batch_size: tuple[int, int] = field(
        default_factory=lambda: tuple(
            CONFIG_READER.get_config_value("attack.tuning.ranges.log_batch_size")
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
    def from_config(cls):
        # config_reader = ConfigReader(config_path=config_path)
        settings_fields = [
            item.name for item in fields(AttackTunerDriverSettings)
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
        paths_fields = [item.name for item in fields(AttackTunerDriverPaths)]
        constructor_kwargs = {
            field_name: PATH_CONFIG_READER.read_path(
                f"attack.tuner_driver.{field_name}"
            )
            for field_name in paths_fields
        }
        return cls(**constructor_kwargs)


class AttackTunerDriverSummary(msgspec.Struct):
    preprocess_id: str
    model_tuning_id: str
    cv_training_id: str
    attack_tuning_id: str
    model_hyperparameters: tuh.X19LSTMHyperParameterSettings
    settings: AttackTunerDriverSettings
    paths: AttackTunerDriverPaths
    study_name: str
    is_continuation: bool
    tuning_ranges: AttackTuningRanges
    model_training_result_dir: str


class AttackTunerDriverSummaryIO(mio.MsgspecIO):
    def __init__(self):
        super().__init__(msgspec_struct_type=AttackTunerDriverSummary)

    @staticmethod
    def enc_hook(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, mds.TrainingCheckpoint):
            return obj.to_storage()
        else:
            raise NotImplementedError(
                f"Encoder does not support objects of type {type(obj)}"
            )

    @staticmethod
    def dec_hook(decode_type: Type, obj: Any) -> Any:
        if decode_type is tuple:
            return tuple(obj)
        if decode_type is torch.Tensor:
            return torch.tensor(obj)
        if decode_type is mds.TrainingCheckpoint:
            return mds.TrainingCheckpointStorage(obj)
        if decode_type is AttackTuningRanges:
            return AttackTuningRanges(**obj)
        if decode_type is Path:
            return Path(obj)


ATTACK_TUNER_DRIVER_SUMMARY_IO = AttackTunerDriverSummaryIO()


@dataclass
class AttackDriverSettings:
    epochs_per_batch: int
    max_num_samples: int
    sample_selection_seed: int
    attack_misclassified_samples: bool

    @classmethod
    def from_config(cls):
        settings_fields = [
            item.name for item in fields(AttackDriverSettings)
        ]
        constructor_kwargs = {
            field_name: CONFIG_READER.get_config_value(
                f"attack.driver_settings.{field_name}"
            )
            for field_name in settings_fields
        }
        return cls(**constructor_kwargs)


@dataclass
class AttackDriverPaths:
    output_dir: Path

    @classmethod
    def from_config(cls):
        paths_fields = [item.name for item in fields(AttackDriverPaths)]
        constructor_kwargs = {
            field_name: PATH_CONFIG_READER.read_path(
                f"attack.attack_driver.{field_name}"
            )
            for field_name in paths_fields
        }
        return cls(**constructor_kwargs)


class AttackDriverSummary(msgspec.Struct):
    preprocess_id: str
    model_tuning_id: str
    cv_training_id: str
    attack_tuning_id: str
    attack_id: str
    settings: AttackDriverSettings
    paths: AttackDriverPaths
    model_hyperparameters: tuh.X19LSTMHyperParameterSettings
    attack_hyperparameters: AttackHyperParameterSettings


class AttackDriverSummaryIO(mio.MsgspecIO):
    def __init__(self):
        super().__init__(msgspec_struct_type=AttackDriverSummary)

    @staticmethod
    def enc_hook(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        else:
            raise NotImplementedError(
                f"Encoder does not support objects of type {type(obj)}"
            )

    @staticmethod
    def dec_hook(decode_type: Type, obj: Any) -> Any:
        if decode_type is torch.Tensor:
            return torch.tensor(obj)
        if decode_type is Path:
            return Path(obj)


ATTACK_DRIVER_SUMMARY_IO = AttackDriverSummaryIO()
