import optuna
import sys
import torch
from pathlib import Path

from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold
from typing import Callable

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfg_path
import lstm_adversarial_attack.config_settings as cfg_set
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.tune_train.hyperparameter_tuner as htu
import lstm_adversarial_attack.tune_train.tuner_helpers as tuh
import lstm_adversarial_attack.x19_mort_general_dataset as xmd


class TunerDriver:
    """
    Instantiates and runs a HyperparameterTuner
    """

    def __init__(
        self,
        device: torch.device,
        collate_fn: Callable = xmd.x19m_collate_fn,
        continue_tuning_dir: Path = None,
        num_folds: int = cfg_set.TUNER_NUM_FOLDS,
        num_cv_epochs: int = cfg_set.TUNER_NUM_CV_EPOCHS,
        epochs_per_fold: int = cfg_set.TUNER_EPOCHS_PER_FOLD,
        fold_class: Callable = StratifiedKFold,
        kfold_random_seed: int = cfg_set.TUNER_KFOLD_RANDOM_SEED,
        cv_mean_metrics_of_interest: tuple[
            str
        ] = cfg_set.TUNER_CV_MEAN_METRICS_OF_INTEREST,
        performance_metric: str = cfg_set.TUNER_PERFORMANCE_METRIC,
        optimization_direction: optuna.study.StudyDirection = cfg_set.TUNER_OPTIMIZATION_DIRECTION,
    ):
        self.device = device
        self.collate_fn = collate_fn
        self.continue_tuning_dir = continue_tuning_dir
        if self.continue_tuning_dir is not None:
            self.output_dir = self.continue_tuning_dir#.parent.parent
        else:
            self.output_dir = rio.create_timestamped_dir(
                parent_path=cfg_path.HYPERPARAMETER_OUTPUT_DIR
            )
        self.tuning_ranges = tuh.X19MLSTMTuningRanges()
        self.num_folds = num_folds
        self.num_cv_epochs = num_cv_epochs
        self.epochs_per_fold = epochs_per_fold
        self.fold_class = fold_class
        self.kfold_random_seed = kfold_random_seed
        self.performance_metric = performance_metric
        self.optimization_direction = optimization_direction
        self.optimization_direction_label = (
            "minimize"
            if optimization_direction == optuna.study.StudyDirection.MINIMIZE
            else "maximize"
        )
        self.pruner = MedianPruner(
            n_startup_trials=cfg_set.TUNER_PRUNER_NUM_STARTUP_TRIALS,
            n_warmup_steps=cfg_set.TUNER_PRUNER_NUM_WARMUP_STEPS,
        )
        self.cv_mean_metrics_of_interest = cv_mean_metrics_of_interest
        self.hyperparameter_sampler = TPESampler()

    def run(self, num_trials: int) -> optuna.Study:
        """
        Runs an optuna study for num_trials
        :param num_trials: number of trials to run
        :return: completed optuna study
        """
        driver_dict_output_path = rio.create_timestamped_filepath(
            parent_path=self.output_dir,
            file_extension="pickle",
            prefix="tuner_driver_dict_",
        )
        rio.ResourceExporter().export(
            resource=self.__dict__, path=driver_dict_output_path
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
            continue_tuning_dir=self.continue_tuning_dir,
            output_dir=self.output_dir,
            kfold_random_seed=self.kfold_random_seed,
            cv_mean_metrics_of_interest=self.cv_mean_metrics_of_interest,
            performance_metric=self.performance_metric,
            optimization_direction=self.optimization_direction,
            pruner=self.pruner,
            hyperparameter_sampler=self.hyperparameter_sampler
        )
        # completed_study = self.tuner.tune(num_trials=num_trials)
        completed_study = tuner.tune(num_trials=num_trials)
        return completed_study

    def __call__(self, num_trials: int) -> optuna.Study:
        completed_study = self.run(num_trials=num_trials)
        return completed_study
