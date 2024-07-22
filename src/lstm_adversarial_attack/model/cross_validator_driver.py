import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import sklearn.model_selection
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.model.cross_validator as cv
import lstm_adversarial_attack.model.model_data_structs as mds
import lstm_adversarial_attack.preprocess.encode_decode_structs as eds
import lstm_adversarial_attack.model.tuner_helpers as tuh
import lstm_adversarial_attack.preprocess.encode_decode as edc
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.x19_mort_general_dataset as xmd


class CrossValidatorDriver:
    """
    Instantiates and runs a CrossValidator.

    Use as isolation layer to avoid modifying CrossValidator code when testing.
    """

    def __init__(
            self,
            preprocess_id: str,
            device: torch.device,
            # dataset: Dataset,
            hyperparameters: tuh.X19LSTMHyperParameterSettings,
            settings: mds.CrossValidatorDriverSettings,
            paths: mds.CrossValidatorDriverPaths,
            tuning_study_name: str = None
    ):

        self.preprocess_id = preprocess_id
        self.device = device
        self.dataset = xmd.X19MGeneralDataset.from_feature_finalizer_output(preprocess_id=preprocess_id)
        self.hyperparameters = hyperparameters
        self.settings = settings
        self.paths = paths
        self.output_dir = self.build_output_dir()
        self.tuning_study_name = tuning_study_name

    def build_output_dir(self) -> Path:
        timestamp = "".join(
            char for char in str(datetime.now()) if char.isdigit()
        )
        output_dir = Path(self.paths.output_dir) / f"cv_training_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

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
            settings=self.settings,
            paths=self.paths,
        )

    def run(self):
        """
        Instantiates and runs CrossValidator
        """

        timestamp = "".join(
            char for char in str(datetime.now()) if char.isdigit()
        )

        summary_output_path = rio.create_timestamped_filepath(
            parent_path=self.output_dir,
            file_extension="json",
            prefix="cross_validator_driver_summary_",
        )
        edc.CrossValidatorDriverSummaryWriter().export(
            obj=self.summary, path=self.output_dir / f"cross_validator_driver_summary_{timestamp}.json"
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
            # cv_output_root_dir=self.paths.output_dir
        )
        cross_validator.run_all_folds()
