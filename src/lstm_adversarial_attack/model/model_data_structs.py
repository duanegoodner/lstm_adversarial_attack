import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


sys.path.append(str(Path(__file__).parent.parent.parent))
from lstm_adversarial_attack.config import ConfigReader, CONFIG_READER


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
        config_reader = ConfigReader(config_path=config_path)
        settings_fields = [
            field.name for field in fields(CrossValidatorDriverSettings)
        ]
        constructor_kwargs = {
            field_name: config_reader.get_config_value(
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
        config_reader = ConfigReader(config_path=config_path)
        paths_fields = [
            field.name for field in fields(CrossValidatorDriverPaths)
        ]
        constructor_kwargs = {
            field_name: config_reader.read_path(
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
            field.name for field in fields(ModelTunerDriverSettings)
        ]
        constructor_kwargs = {
            field_name: CONFIG_READER.get_config_value(
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
        paths_fields = [field.name for field in fields(ModelTunerDriverPaths)]
        constructor_kwargs = {
            field_name: CONFIG_READER.read_path(
                f"model.tuner_driver.{field_name}"
            )
            for field_name in paths_fields
        }
        return cls(**constructor_kwargs)
