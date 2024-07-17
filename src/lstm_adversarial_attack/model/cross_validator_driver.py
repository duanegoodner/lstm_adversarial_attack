import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable

import sklearn.model_selection
import torch
from torch.utils.data import Dataset

from lstm_adversarial_attack.config import ConfigReader

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.model.cross_validator as cv
import lstm_adversarial_attack.model.tuner_helpers as tuh
import lstm_adversarial_attack.x19_mort_general_dataset as xmd


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
                f"model.tuner_driver.{field_name}")
            for field_name in paths_fields}
        return cls(**constructor_kwargs)


class CrossValidatorDriver:
    """
    Instantiates and runs a CrossValidator.

    Use as isolation layer to avoid modifying CrossValidator code when testing.
    """

    def __init__(
            self,
            device: torch.device,
            dataset: Dataset,
            hyperparameters: tuh.X19LSTMHyperParameterSettings,
            settings: CrossValidatorDriverSettings,
            paths: CrossValidatorDriverPaths,
            tuning_study_name: str = None
    ):
        self.device = device
        self.dataset = dataset
        self.hyperparameters = hyperparameters
        self.settings = settings
        self.paths = paths
        self.tuning_study_name = tuning_study_name

    @property
    def fold_class(self) -> sklearn.model_selection.BaseCrossValidator:
        return getattr(sklearn.model_selection, self.settings.fold_class_name)

    @property
    def collate_fn(self) -> Callable:
        return getattr(xmd, self.settings.collate_fn_name)

    def run(self):
        """
        Instantiates and runs CrossValidator
        """
        cross_validator = cv.CrossValidator(
            device=self.device,
            dataset=self.dataset,
            hyperparameter_settings=self.hyperparameters,
            num_folds=self.settings.num_folds,
            epochs_per_fold=self.settings.epochs_per_fold,
            eval_interval=self.settings.eval_interval,
            kfold_random_seed=self.settings.kfold_random_seed,
            fold_class=self.fold_class,
            collate_fn=self.collate_fn,
            single_fold_eval_fraction=self.settings.single_fold_eval_fraction,
            cv_output_root_dir=self.paths.output_dir
        )
        cross_validator.run_all_folds()
