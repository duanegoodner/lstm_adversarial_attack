import sys
from dataclasses import dataclass, fields
from pathlib import Path

from lstm_adversarial_attack.config import ConfigReader

sys.path.append(str(Path(__file__).parent.parent.parent))


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
        settings_fields = [field.name for field in
                           fields(CrossValidatorDriverSettings)]
        constructor_kwargs = {field_name: config_reader.get_config_value(
            f"model.cv_driver_settings.{field_name}") for field_name in
            settings_fields}
        return cls(**constructor_kwargs)


@dataclass
class CrossValidatorDriverPaths:
    output_dir: str

    @classmethod
    def from_config(cls, config_path: Path = None):
        config_reader = ConfigReader(config_path=config_path)
        paths_fields = [field.name for field in fields(CrossValidatorDriverPaths)]
        constructor_kwargs = {
            field_name: config_reader.read_path(
                f"model.cv_driver.{field_name}")
            for field_name in paths_fields}
        return cls(**constructor_kwargs)