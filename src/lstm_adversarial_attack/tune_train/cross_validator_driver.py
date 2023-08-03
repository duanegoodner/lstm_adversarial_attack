import sys
import torch
from pathlib import Path
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfg_path
import lstm_adversarial_attack.config_settings as lcs
import lstm_adversarial_attack.data_structures as ds
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.tune_train.cross_validator as cv
import lstm_adversarial_attack.tune_train.tuner_helpers as tuh


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
        epochs_per_fold: int = lcs.CV_DRIVER_EPOCHS_PER_FOLD,
        num_folds: int = lcs.CV_DRIVER_NUM_FOLDS,
        eval_interval: int = lcs.CV_DRIVER_EVAL_INTERVAL,
    ):
        self.device = device
        self.dataset = dataset
        self.hyperparameters = hyperparameters
        self.epochs_per_fold = epochs_per_fold
        self.num_folds = num_folds
        self.eval_interval = eval_interval

    @classmethod
    def from_study_path(
        cls,
        device: torch.device,
        dataset: Dataset,
        study_path: Path,
        num_folds: int = lcs.CV_DRIVER_NUM_FOLDS,
        epochs_per_fold: int = lcs.CV_DRIVER_EPOCHS_PER_FOLD
    ):
        """
        Creates CrossValidationDriver from path to optuna.Study pickle file
        :param device: device to run on
        :param dataset: full dataset for cross validation
        :param study_path: path to optuna.Study pickle file
        :param num_folds: number cross-validation folds
        :param epochs_per_fold: number training epochs to run on each fold
        """
        study = rio.ResourceImporter().import_pickle_to_object(path=study_path)
        hyperparameters = tuh.X19LSTMHyperParameterSettings(
            **study.best_params
        )
        return cls(
            device=device,
            dataset=dataset,
            hyperparameters=hyperparameters,
            num_folds=num_folds,
            epochs_per_fold=epochs_per_fold
        )

    def run(self):
        """
        Instantiates and runs CrossValidator
        """
        cross_validator = cv.CrossValidator(
            device=self.device,
            dataset=self.dataset,
            hyperparameter_settings=self.hyperparameters,
            num_folds=self.num_folds,
            epochs_per_fold=self.epochs_per_fold,
            eval_interval=self.eval_interval,
        )
        cross_validator.run_all_folds()
