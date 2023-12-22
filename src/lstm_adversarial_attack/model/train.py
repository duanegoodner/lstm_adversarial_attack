import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.config_settings as cfg_set
import lstm_adversarial_attack.data_structures as ds
import lstm_adversarial_attack.gpu_helpers as gh
import lstm_adversarial_attack.path_searches as ps
import lstm_adversarial_attack.model.cross_validator_driver as cvd
import lstm_adversarial_attack.model.tuner_helpers as tuh
import lstm_adversarial_attack.tuning_db.tuning_studies_database as tsd
import lstm_adversarial_attack.x19_mort_general_dataset as xmd


def main(
    study_name: str = None,
    num_folds: int = None,
    epochs_per_fold: int = None,
) -> dict[int, ds.TrainEvalLogPair]:
    # use if is None syntax (instead of default args) for CLI integration
    if study_name is None:
        study_name = tsd.MODEL_TUNING_DB.get_latest_study().study_name

    hyperparams_dict = tsd.MODEL_TUNING_DB.get_best_params(
        study_name=study_name
    )
    hyperparameters = tuh.X19LSTMHyperParameterSettings(
        **hyperparams_dict
    )

    if num_folds is None:
        num_folds = cfg_set.CV_DRIVER_NUM_FOLDS
    if epochs_per_fold is None:
        epochs_per_fold = cfg_set.CV_DRIVER_EPOCHS_PER_FOLD
    assert num_folds > 0

    cur_device = gh.get_device()

    cv_driver = cvd.CrossValidatorDriver(
        device=cur_device,
        dataset=xmd.X19MGeneralDataset.from_feature_finalizer_output(),
        hyperparameters=hyperparameters,
        epochs_per_fold=epochs_per_fold,
        num_folds=num_folds,
        tuning_study_name=study_name
    )

    return cv_driver.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--study_name",
        type=str,
        action="store",
        nargs="?",
        help=(
            "Name of the optuna study in RDB to obtain hyperparameters from"
        ),
    )
    parser.add_argument(
        "-f",
        "--num_folds",
        type=int,
        action="store",
        nargs="?",
        help=(
            "Number of cross-validation folds. Defaults to "
            "config_settings.CV_DRIVER_NUM_FOLDS"
        ),
    )

    parser.add_argument(
        "-e",
        "--epochs_per_fold",
        type=int,
        action="store",
        nargs="?",
        help=(
            "Number of epochs per fold. Defaults to "
            "config_settings.CV_DRIVER_EPOCHS_PER_FOLD"
        ),
    )

    args_namespace = parser.parse_args()

    all_fold_results = main(**args_namespace.__dict__)
