import argparse
import json
import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.config_settings as cfg_set
import lstm_adversarial_attack.data_structures as ds
import lstm_adversarial_attack.gpu_helpers as gh
import lstm_adversarial_attack.path_searches as ps
import lstm_adversarial_attack.tune_train.cross_validator_driver as cvd
import lstm_adversarial_attack.tune_train.tuner_helpers as tuh
import lstm_adversarial_attack.x19_mort_general_dataset as xmd


def main(
    # study_path: str = None,
    hyperparameters_json_path: str = None,
    num_folds: int = None,
    epochs_per_fold: int = None,
) -> dict[int, ds.TrainEvalLogPair]:
    # use if is None syntax (instead of default args) for CLI integration
    if hyperparameters_json_path is None:
        hyperparameters_json_path = (
            ps.latest_modified_file_with_name_condition(
                component_string="best_trial_info.json",
                root_dir=cfg_paths.HYPERPARAMETER_OUTPUT_DIR,
            )
        )
    with Path(hyperparameters_json_path).open(mode="r") as input_file:
        hyperparams_dict = json.load(input_file)

    hyperparameters = tuh.X19LSTMHyperParameterSettings(
        **hyperparams_dict["hyperparameters"]
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
    )

    return cv_driver.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-j",
        "--hyperparameters_json_path",
        type=str,
        action="store",
        nargs="?",
        help=(
            "Path to an json file containing hyperparameters to use during "
            "model training. Format must match that of file output by"
            "HyperParameterTuner.export_best_trial_info(). If "
            "not specified, the most recently modified file named "
            "'best_trial_info.json' under path specified by "
            "config_paths.HYPERPARAMETER_OUTPUT_DIR will be used."
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
