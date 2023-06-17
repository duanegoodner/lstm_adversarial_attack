import sys

import optuna
import torch
from pathlib import Path
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_settings as lcs
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.tune_train.cross_validator as cv
import lstm_adversarial_attack.tune_train.tuner_helpers as tuh


class CrossValidatorDriver:
    def __init__(
        self,
        device: torch.device,
        dataset: Dataset,
        hyperparameters: tuh.X19LSTMHyperParameterSettings,
        epochs_per_fold: int = lcs.CV_DRIVER_EPOCHS_PER_FOLD,
        num_folds: int = lcs.CV_DRIVER_NUM_FOLDS,
        eval_interval: int = lcs.CV_DRIVER_EVAL_INTERVAL,
        evals_per_checkpoint: int = lcs.CV_DRIVER_EVALS_PER_CHECKPOINT,
    ):
        self.device = device
        self.dataset = dataset
        self.hyperparameters = hyperparameters
        self.epochs_per_fold = epochs_per_fold
        self.num_folds = num_folds
        self.eval_interval = eval_interval
        self.evals_per_checkpoint = evals_per_checkpoint

    @classmethod
    def from_study_path(
        cls, device: torch.device, dataset: Dataset, study_path: Path
    ):
        study = rio.ResourceImporter().import_pickle_to_object(path=study_path)
        hyperparameters = tuh.X19LSTMHyperParameterSettings(
            **study.best_params
        )
        return cls(
            device=device, dataset=dataset, hyperparameters=hyperparameters
        )

    def run(self):
        cross_validator = cv.CrossValidator(
            device=self.device,
            dataset=self.dataset,
            hyperparameter_settings=self.hyperparameters,
            num_folds=self.num_folds,
            epochs_per_fold=self.epochs_per_fold,
            eval_interval=self.eval_interval,
            evals_per_checkpoint=self.evals_per_checkpoint
        )
        cross_validator.run_all_folds()
