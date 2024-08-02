import sys
from functools import cached_property
from pathlib import Path
from typing import Callable

import optuna
import sklearn.model_selection
import torch

from lstm_adversarial_attack.config import CONFIG_READER

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.model.model_data_structs as mds
import lstm_adversarial_attack.model.model_tuner as htu
import lstm_adversarial_attack.model.tuner_helpers as tuh
import lstm_adversarial_attack.tuning_db.tuning_studies_database as tsd
import lstm_adversarial_attack.dataset.x19_mort_general_dataset as xmd


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


class ModelTunerDriver:
    """
    Instantiates and runs a HyperparameterTuner
    """

    def __init__(
        self,
        device: torch.device,
        settings: mds.ModelTunerDriverSettings,
        paths: mds.ModelTunerDriverPaths,
        preprocess_id: str,
        model_tuning_id: str,
    ):
        self.device = device
        self.settings = settings
        self.paths = paths
        self.preprocess_id = preprocess_id
        self.model_tuning_id = model_tuning_id
        self.study_name = f"model_tuning_{self.model_tuning_id}"
        self.output_dir = Path(self.paths.output_dir) / self.model_tuning_id
        self.has_pre_existing_rdb_output = has_rdb_output(
            study_name=self.study_name, storage=self.db.storage
        )
        self.has_pre_existing_local_output = self.output_dir.exists()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_model_tuning_id(cls, model_tuning_id: str, device: torch.device):
        tuning_output_root = Path(
            CONFIG_READER.read_path("model.tuner_driver.output_dir")
        )

        tuner_driver_summary = mds.TUNER_DRIVER_SUMMARY_IO.import_to_struct(
            path=tuning_output_root
                 / model_tuning_id
                 / f"tuner_driver_summary_{model_tuning_id}.json"
        )

        return cls(
            device=device,
            model_tuning_id=model_tuning_id,
            settings=tuner_driver_summary.settings,
            paths=tuner_driver_summary.paths,
            preprocess_id=tuner_driver_summary.preprocess_id,
        )

    @property
    def collate_fn(self) -> Callable:
        return getattr(xmd, self.settings.collate_fn_name)

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
            **self.settings.pruner_kwargs
        )

    @property
    def hyperparameter_sampler(self) -> optuna.samplers.BaseSampler:
        return getattr(optuna.samplers, self.settings.sampler_name)(
            **self.settings.sampler_kwargs
        )

    @cached_property
    def db(self) -> tsd.OptunaDatabase:
        db_dotenv_info = tsd.get_db_dotenv_info(
            db_name_var=self.settings.db_env_var_name
        )
        return tsd.OptunaDatabase(**db_dotenv_info)

    @property
    def summary(self) -> mds.TunerDriverSummary:
        return mds.TunerDriverSummary(
            preprocess_id=self.preprocess_id,
            model_tuning_id=self.model_tuning_id,
            settings=self.settings,
            paths=self.paths,
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
            summary_output_path = (
                self.output_dir
                / f"tuner_driver_summary_{self.model_tuning_id}.json"
            )

            mds.TUNER_DRIVER_SUMMARY_IO.export(obj=self.summary, path=summary_output_path)

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
            dataset=xmd.X19MGeneralDataset.from_feature_finalizer_output(
                preprocess_id=self.preprocess_id
            ),
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
        completed_study = tuner.tune(num_trials=num_trials)
        return completed_study

    def __call__(self, num_trials: int) -> optuna.Study:
        completed_study = self.run(num_trials=num_trials)
        return completed_study
