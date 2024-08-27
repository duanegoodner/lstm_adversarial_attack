import sys
from pathlib import Path
from typing import Callable

import optuna
import sklearn.model_selection
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.dataset.x19_mort_general_dataset as xmd
import lstm_adversarial_attack.model.cross_validator as cv
import lstm_adversarial_attack.model.model_data_structs as mds
import lstm_adversarial_attack.model.tuner_helpers as tuh
import lstm_adversarial_attack.tuning_db.tuning_studies_database as tsd
import lstm_adversarial_attack.utils.redirect_output as rdo
from lstm_adversarial_attack.config.read_write import (
    CONFIG_READER,
    PATH_CONFIG_READER,
)


class CrossValidatorDriver:
    """
    Instantiates and runs a CrossValidator.

    Use as isolation layer to avoid modifying CrossValidator code when testing.
    """

    def __init__(
        self,
        model_tuning_id: str,
        cv_training_id: str,
        device: torch.device,
        redirect_terminal_output: bool,
        model_tuning_trial_number: int = None,
    ):

        self.model_tuning_id = model_tuning_id
        self.cv_training_id = cv_training_id
        self.device = device
        self.redirect_terminal_output = redirect_terminal_output
        if model_tuning_trial_number is None:
            model_tuning_trial_number = self.tuning_study.best_trial.number
        self.model_tuning_trial_number = model_tuning_trial_number
        self.dataset = xmd.X19MGeneralDataset.from_feature_finalizer_output(
            preprocess_id=self.preprocess_id
        )
        self.settings = mds.CrossValidatorDriverSettings.from_config()
        self.paths = mds.CrossValidatorDriverPaths.from_config()
        self.output_dir = self.build_output_dir()

    @property
    def model_tuner_driver_summary(self) -> mds.CrossValidatorDriverSummary:
        model_tuning_output_root = Path(
            PATH_CONFIG_READER.read_path("model.tuner_driver.output_dir")
        )
        return mds.TUNER_DRIVER_SUMMARY_IO.import_to_struct(
            path=model_tuning_output_root
            / self.model_tuning_id
            / f"tuner_driver_summary_{self.model_tuning_id}.json"
        )

    @property
    def preprocess_id(self) -> str:
        return self.model_tuner_driver_summary.preprocess_id

    def build_output_dir(self) -> Path:
        output_dir = Path(self.paths.output_dir) / f"{self.cv_training_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @property
    def tuning_study_name(self) -> str:
        return f"model_tuning_{self.model_tuning_id}"

    @property
    def tuning_study(self) -> optuna.Study:
        return tsd.MODEL_TUNING_DB.get_study(study_name=self.tuning_study_name)

    @property
    def hyperparameters(self) -> tuh.X19LSTMHyperParameterSettings:
        hyperparams_dict = self.tuning_study.trials[
            self.model_tuning_trial_number
        ].params
        return tuh.X19LSTMHyperParameterSettings(**hyperparams_dict)

    @property
    def fold_class(self) -> sklearn.model_selection.BaseCrossValidator:
        return getattr(sklearn.model_selection, self.settings.fold_class_name)

    @property
    def collate_fn(self) -> Callable:
        return getattr(xmd, self.settings.collate_fn_name)

    @property
    def summary(self) -> mds.CrossValidatorDriverSummary:
        return mds.CrossValidatorDriverSummary(
            preprocess_id=self.preprocess_id,
            model_tuning_id=self.model_tuning_id,
            model_tuning_trial_number=self.model_tuning_trial_number,
            cv_training_id=self.cv_training_id,
            model_hyperparameters=self.hyperparameters,
            settings=self.settings,
            paths=self.paths,
        )

    def run(self):
        """
        Instantiates and runs CrossValidator
        """

        CONFIG_READER.record_full_config(root_dir=self.output_dir)
        PATH_CONFIG_READER.record_full_config(root_dir=self.output_dir)

        mds.CROSS_VALIDATOR_DRIVER_SUMMARY_IO.export(
            obj=self.summary,
            path=self.output_dir
            / f"cross_validator_driver_summary_{self.cv_training_id}.json",
        )

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
            output_dir=self.output_dir,
        )

        log_file_fid = None
        if self.redirect_terminal_output:
            log_file_path = (
                Path(
                    PATH_CONFIG_READER.read_path(
                        "redirected_output.model_training"
                    )
                )
                / f"{self.cv_training_id}.log"
            )
            log_file_fid = rdo.set_redirection(
                log_file_path=log_file_path, include_optuna=False
            )

        cross_validator.run_all_folds()

        if self.redirect_terminal_output:
            log_file_fid.close()
