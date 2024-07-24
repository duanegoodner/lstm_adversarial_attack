import sys
from pathlib import Path
from typing import Callable

import sklearn.model_selection
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.model.cross_validator as cv
import lstm_adversarial_attack.model.model_data_structs as mds
import lstm_adversarial_attack.model.tuner_helpers as tuh
import lstm_adversarial_attack.preprocess.encode_decode as edc
import lstm_adversarial_attack.preprocess.encode_decode_structs as eds
import lstm_adversarial_attack.x19_mort_general_dataset as xmd
from lstm_adversarial_attack.config import CONFIG_READER
import lstm_adversarial_attack.tuning_db.tuning_studies_database as tsd


class CrossValidatorDriver:
    """
    Instantiates and runs a CrossValidator.

    Use as isolation layer to avoid modifying CrossValidator code when testing.
    """

    def __init__(
            self,
            preprocess_id: str,
            cv_training_id: str,
            device: torch.device,
            hyperparameters: tuh.X19LSTMHyperParameterSettings,
            settings: mds.CrossValidatorDriverSettings,
            paths: mds.CrossValidatorDriverPaths,
            tuning_study_name: str = None
    ):

        self.preprocess_id = preprocess_id
        self.cv_training_id = cv_training_id
        self.device = device
        self.dataset = xmd.X19MGeneralDataset.from_feature_finalizer_output(preprocess_id=preprocess_id)
        self.hyperparameters = hyperparameters
        self.settings = settings
        self.paths = paths
        self.output_dir = self.build_output_dir()
        self.tuning_study_name = tuning_study_name

    def build_output_dir(self) -> Path:
        output_dir = Path(self.paths.output_dir) / f"{self.cv_training_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @classmethod
    def from_model_tuning_id(cls, model_tuning_id: str, cv_training_id: str, device: torch.device):
        study_name = f"model_tuning_{model_tuning_id}"
        hyperparams_dict = tsd.MODEL_TUNING_DB.get_best_params(
            study_name=study_name
        )
        hyperparameters = tuh.X19LSTMHyperParameterSettings(**hyperparams_dict)

        model_tuning_output_root = Path(
            CONFIG_READER.read_path("model.tuner_driver.output_dir")
        )
        model_tuner_driver_summary = edc.TunerDriverSummaryReader().import_struct(
            path=model_tuning_output_root
                 / model_tuning_id
                 / f"tuner_driver_summary_{model_tuning_id}.json"
        )

        return cls(
            preprocess_id=model_tuner_driver_summary.preprocess_id,
            cv_training_id=cv_training_id,
            device=device,
            hyperparameters=hyperparameters,
            settings=mds.CrossValidatorDriverSettings.from_config(),
            paths=mds.CrossValidatorDriverPaths.from_config(),
            tuning_study_name=study_name,
        )

    @property
    def fold_class(self) -> sklearn.model_selection.BaseCrossValidator:
        return getattr(sklearn.model_selection, self.settings.fold_class_name)

    @property
    def collate_fn(self) -> Callable:
        return getattr(xmd, self.settings.collate_fn_name)

    @property
    def summary(self) -> eds.CrossValidatorDriverSummary:
        return eds.CrossValidatorDriverSummary(
            preprocess_id=self.preprocess_id,
            tuning_study_name=self.tuning_study_name,
            cv_training_id=self.cv_training_id,
            model_hyperparameters=self.hyperparameters,
            settings=self.settings,
            paths=self.paths,
        )

    def run(self):
        """
        Instantiates and runs CrossValidator
        """

        edc.CrossValidatorDriverSummaryWriter().export(
            obj=self.summary, path=self.output_dir / f"cross_validator_driver_summary_{self.cv_training_id}.json"
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
        cross_validator.run_all_folds()
