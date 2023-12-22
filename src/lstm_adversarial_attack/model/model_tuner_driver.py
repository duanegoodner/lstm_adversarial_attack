import sys
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any

import optuna
import sklearn.model_selection
import torch

from lstm_adversarial_attack.config import ConfigReader

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfp
import lstm_adversarial_attack.config_settings as cfg_set
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


# def has_local_output(study_name: str) -> bool:
#     """
#     Determines if local output exists for study named study_name
#     :param study_name: name of study
#     :return:
#     """
#     return (cfp.HYPERPARAMETER_OUTPUT_DIR / study_name).exists()


@dataclass
class ModelTunerDriverSettings:
    num_folds: int
    num_cv_epochs: int
    epochs_per_fold: int
    kfold_random_seed: int
    cv_mean_metrics_of_interest: list[str]
    performance_metric: str
    optimization_direction_label: str
    hyperparameter_output_dir: Path

    # @classmethod
    # def from_config(cls, config_path: Path = None):
    #     config_reader = ConfigReader(config_path=config_path)
    #     return cls(
    #         num_folds=config_reader.
    #     )


class ModelTunerDriver:
    """
    Instantiates and runs a HyperparameterTuner
    """

    def __init__(
        self,
        device: torch.device,
        # settings: ModelTunerDriverSettings,
        db_env_var_name: str = "MODEL_TUNING_DB_NAME",
        study_name: str = None,
        collate_fn_name: str = "x19m_collate_fn",
        tuning_ranges: tuh.X19MLSTMTuningRanges = None,
        num_folds: int = cfg_set.TUNER_NUM_FOLDS,
        num_cv_epochs: int = cfg_set.TUNER_NUM_CV_EPOCHS,
        epochs_per_fold: int = cfg_set.TUNER_EPOCHS_PER_FOLD,
        fold_class_name: str = "StratifiedKFold",
        kfold_random_seed: int = cfg_set.TUNER_KFOLD_RANDOM_SEED,
        cv_mean_metrics_of_interest: tuple[
            str, ...
        ] = cfg_set.TUNER_CV_MEAN_TENSORBOARD_METRICS,
        performance_metric: str = cfg_set.TUNER_PERFORMANCE_METRIC,
        optimization_direction_label: str = cfg_set.TUNER_OPTIMIZATION_DIRECTION,
        pruner_name: str = "MedianPruner",
        pruner_kwargs: dict[str, Any] = None,
        sampler_name: str = "TPESampler",
        sampler_kwargs: dict[str, Any] = None,
    ):
        self.device = device
        # self.settings = settings
        self.db_env_var_name = db_env_var_name
        self.collate_fn = getattr(xmd, collate_fn_name)
        if study_name is None:
            study_name = self.build_study_name()
        self.study_name = study_name
        self.has_pre_existing_rdb_output = has_rdb_output(
            study_name=self.study_name, storage=self.db.storage
        )
        self.output_dir = cfp.HYPERPARAMETER_OUTPUT_DIR / self.study_name
        self.has_pre_existing_local_output = self.output_dir.exists()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if tuning_ranges is None:
            tuning_ranges = tuh.X19MLSTMTuningRanges()
        self.tuning_ranges = tuning_ranges
        self.num_folds = num_folds
        self.num_cv_epochs = num_cv_epochs
        self.epochs_per_fold = epochs_per_fold
        self.fold_class = getattr(sklearn.model_selection, fold_class_name)
        self.kfold_random_seed = kfold_random_seed
        self.performance_metric = performance_metric
        self.optimization_direction_label = optimization_direction_label
        self.optimization_direction = self.get_optimization_direction()
        if pruner_kwargs is None:
            pruner_kwargs = {
                "n_startup_trials": cfg_set.TUNER_PRUNER_NUM_STARTUP_TRIALS,
                "n_warmup_steps": cfg_set.TUNER_PRUNER_NUM_WARMUP_STEPS,
            }
        self.pruner_kwargs = pruner_kwargs
        self.pruner = self.get_pruner(pruner_name=pruner_name)
        self.cv_mean_metrics_of_interest = cv_mean_metrics_of_interest
        if sampler_kwargs is None:
            sampler_kwargs = {}
        self.sampler_kwargs = sampler_kwargs
        self.hyperparameter_sampler = self.get_sampler(
            sampler_name=sampler_name
        )

    @staticmethod
    def build_study_name() -> str:
        timestamp = "".join(
            char for char in str(datetime.now()) if char.isdigit()
        )
        return f"model_tuning_{timestamp}"

    def validate_optimization_direction_label(self):
        assert (
            self.optimization_direction_label == "minimize"
            or self.optimization_direction_label == "maximize"
        )

    def get_optimization_direction(self) -> optuna.study.StudyDirection:
        self.validate_optimization_direction_label()
        return (
            optuna.study.StudyDirection.MINIMIZE
            if self.optimization_direction_label == "minimize"
            else optuna.study.StudyDirection.MAXIMIZE
        )

    def get_pruner(self, pruner_name: str) -> optuna.pruners.BasePruner:
        return getattr(optuna.pruners, pruner_name)(**self.pruner_kwargs)

    def get_sampler(self, sampler_name: str) -> optuna.samplers.BaseSampler:
        return getattr(optuna.samplers, sampler_name)(**self.sampler_kwargs)

    @cached_property
    def db(self) -> tsd.OptunaDatabase:
        db_dotenv_info = tsd.get_db_dotenv_info(
            db_name_var=self.db_env_var_name
        )
        return tsd.OptunaDatabase(**db_dotenv_info)

    @property
    def summary(self) -> eds.TunerDriverSummary:
        return eds.TunerDriverSummary(
            collate_fn_name=self.collate_fn.__name__,
            db_env_var_name=self.db_env_var_name,
            study_name=self.study_name,
            # TODO should continuation check be for local, RDB, or both?
            is_continuation=self.has_pre_existing_local_output,
            cv_mean_metrics_of_interest=self.cv_mean_metrics_of_interest,
            device_name=str(self.device),
            epochs_per_fold=self.epochs_per_fold,
            fold_class_name=self.fold_class.__name__,
            output_dir=str(self.output_dir),
            sampler_name=self.hyperparameter_sampler.__class__.__name__,
            sampler_kwargs=self.sampler_kwargs,
            kfold_random_seed=self.kfold_random_seed,
            num_cv_epochs=self.num_cv_epochs,
            num_folds=self.num_folds,
            optimization_direction_label=self.optimization_direction_label,
            performance_metric=self.performance_metric,
            pruner_name=self.pruner.__class__.__name__,
            tuning_ranges=self.tuning_ranges,
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
            direction=self.optimization_direction_label,
            sampler=self.hyperparameter_sampler,
            pruner=self.pruner,
        )

        tuner = htu.HyperParameterTuner(
            device=self.device,
            dataset=xmd.X19MGeneralDataset.from_feature_finalizer_output(),
            collate_fn=self.collate_fn,
            tuning_ranges=self.tuning_ranges,
            num_folds=self.num_folds,
            num_cv_epochs=self.num_cv_epochs,
            epochs_per_fold=self.epochs_per_fold,
            fold_class=self.fold_class,
            output_dir=self.output_dir,
            kfold_random_seed=self.kfold_random_seed,
            cv_mean_metrics_of_interest=self.cv_mean_metrics_of_interest,
            performance_metric=self.performance_metric,
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
