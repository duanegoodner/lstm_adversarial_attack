import sys
from dataclasses import dataclass, fields
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, Callable

import optuna
import sklearn.model_selection
import torch

from lstm_adversarial_attack.config import ConfigReader

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.preprocess.encode_decode as edc
import lstm_adversarial_attack.preprocess.encode_decode_structs as eds
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.model.model_tuner as htu
import lstm_adversarial_attack.model.tuner_helpers as tuh
import lstm_adversarial_attack.tuning_db.tuning_studies_database as tsd
import lstm_adversarial_attack.x19_mort_general_dataset as xmd


def has_rdb_output(
        study_name: str, storage: optuna.storages.RDBStorage
) -> bool:
    """
    Determines if output exists for study named study_name in storage.
    :param study_name: name of study
    :param storage: RDB storage where we look for study
    :return: bool indicating presence of output
    """
    existing_study_summaries = optuna.study.get_all_study_summaries(
        storage=storage
    )
    existing_study_names = [
        item.study_name for item in existing_study_summaries
    ]
    return study_name in existing_study_names


@dataclass
class ModelTunerDriverSettings:
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
    def from_config(cls, config_path: Path = None):
        config_reader = ConfigReader(config_path=config_path)
        settings_fields = [field.name for field in
                           fields(ModelTunerDriverSettings)]
        constructor_kwargs = {field_name: config_reader.get_config_value(
            f"model.tuner_driver.{field_name}") for field_name in
            settings_fields}
        return cls(**constructor_kwargs)


@dataclass
class ModelTunerDriverPaths:
    output_dir: str

    @classmethod
    def from_config(cls, config_path: Path = None):
        config_reader = ConfigReader(config_path=config_path)
        paths_fields = [field.name for field in fields(ModelTunerDriverPaths)]
        constructor_kwargs = {
            field_name: config_reader.read_path(
                f"model.tuner_driver.{field_name}")
            for field_name in paths_fields}
        return cls(**constructor_kwargs)


class ModelTunerDriver:
    """
    Instantiates and runs a HyperparameterTuner
    """
    def __init__(
            self,
            device: torch.device,
            settings: ModelTunerDriverSettings,
            paths: ModelTunerDriverPaths,
            study_name: str = None,
    ):
        self.device = device
        self.settings = settings
        self.paths = paths
        if study_name is None:
            study_name = self.build_study_name()
        self.study_name = study_name
        self.has_pre_existing_rdb_output = has_rdb_output(
            study_name=self.study_name, storage=self.db.storage
        )
        self.has_pre_existing_local_output = self.output_dir.exists()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def build_study_name() -> str:
        timestamp = "".join(
            char for char in str(datetime.now()) if char.isdigit()
        )
        return f"model_tuning_{timestamp}"

    @property
    def collate_fn(self) -> Callable:
        return getattr(xmd, self.settings.collate_fn_name)

    @property
    def output_dir(self) -> Path:
        return Path(self.paths.output_dir) / self.study_name

    @property
    def tuning_ranges(self) -> tuh.X19MLSTMTuningRanges:
        return tuh.X19MLSTMTuningRanges(**self.settings.tuning_ranges)

    @property
    def fold_class(self) -> sklearn.model_selection.BaseCrossValidator:
        return getattr(sklearn.model_selection, self.settings.fold_class_name)

    def validate_optimization_direction_label(self):
        assert (
                self.settings.optimization_direction_label == "minimize"
                or self.settings.optimization_direction_label == "maximize"
        )

    @property
    def optimization_direction(self) -> optuna.study.StudyDirection:
        self.validate_optimization_direction_label()
        return (
            optuna.study.StudyDirection.MINIMIZE
            if self.settings.optimization_direction_label == "minimize"
            else optuna.study.StudyDirection.MAXIMIZE
        )

    @property
    def pruner(self) -> optuna.pruners.BasePruner:
        return getattr(optuna.pruners, self.settings.pruner_name)(
            **self.settings.pruner_kwargs)

    @property
    def hyperparameter_sampler(self) -> optuna.samplers.BaseSampler:
        return getattr(optuna.samplers, self.settings.sampler_name)(
            **self.settings.sampler_kwargs)

    @cached_property
    def db(self) -> tsd.OptunaDatabase:
        db_dotenv_info = tsd.get_db_dotenv_info(
            db_name_var=self.settings.db_env_var_name
        )
        return tsd.OptunaDatabase(**db_dotenv_info)

    @property
    def summary(self) -> eds.TunerDriverSummary:
        return eds.TunerDriverSummary(
            settings=self.settings.__dict__,
            paths=self.paths.__dict__,
            study_name=self.study_name,
            # TODO should continuation check be for local, RDB, or both?
            is_continuation=self.has_pre_existing_local_output,
            device_name=str(self.device),
        )

    def run(self, num_trials: int) -> optuna.Study:
        """
        Runs an optuna study for num_trials
        :param num_trials: number of trials to run
        :return: completed optuna study
        """
        if not self.summary.is_continuation:
            summary_output_path = rio.create_timestamped_filepath(
                parent_path=self.output_dir,
                file_extension="json",
                prefix="tuner_driver_summary_",
            )
            edc.TunerDriverSummaryWriter().export(
                obj=self.summary, path=summary_output_path
            )

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.db.storage,
            load_if_exists=True,
            direction=self.settings.optimization_direction_label,
            sampler=self.hyperparameter_sampler,
            pruner=self.pruner,
        )

        tuner = htu.HyperParameterTuner(
            device=self.device,
            dataset=xmd.X19MGeneralDataset.from_feature_finalizer_output(),
            collate_fn=self.collate_fn,
            tuning_ranges=self.tuning_ranges,
            num_folds=self.settings.num_folds,
            num_cv_epochs=self.settings.num_cv_epochs,
            epochs_per_fold=self.settings.epochs_per_fold,
            fold_class=self.fold_class,
            output_dir=Path(self.output_dir),
            kfold_random_seed=self.settings.kfold_random_seed,
            cv_mean_metrics_of_interest=self.settings.cv_mean_tensorboard_metrics,
            performance_metric=self.settings.performance_metric,
            optimization_direction=self.optimization_direction,
            pruner=self.pruner,
            hyperparameter_sampler=self.hyperparameter_sampler,
            study=study,
        )
        # completed_study = self.tuner.tune(num_trials=num_trials)
        completed_study = tuner.tune(num_trials=num_trials)
        return completed_study

    def __call__(self, num_trials: int) -> optuna.Study:
        completed_study = self.run(num_trials=num_trials)
        return completed_study
