import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config as config
import lstm_adversarial_attack.data_structures as ds
import lstm_adversarial_attack.gpu_helpers as gh
import lstm_adversarial_attack.model.cross_validator_driver as cvd
import lstm_adversarial_attack.model.tuner_helpers as tuh
import lstm_adversarial_attack.tuning_db.tuning_studies_database as tsd
import lstm_adversarial_attack.x19_mort_general_dataset as xmd


def main(
        study_name: str = None,
        num_folds: int = None,
        epochs_per_fold: int = None,
) -> dict[int, ds.TrainEvalLogPair]:

    if study_name is None:
        study_name = tsd.MODEL_TUNING_DB.get_latest_study().study_name

    # get hyperparameters from database
    hyperparams_dict = tsd.MODEL_TUNING_DB.get_best_params(
        study_name=study_name
    )
    hyperparameters = tuh.X19LSTMHyperParameterSettings(
        **hyperparams_dict
    )

    # build settings object for CrossValidatorDriver
    config_reader = config.ConfigReader()
    # config_settings = config_reader.get_config_value("model.cv_driver_settings")
    # cv_driver_settings = cvd.CrossValidatorDriverSettings(**config_settings)

    cv_driver_settings = cvd.CrossValidatorDriverSettings.from_config()

    if num_folds is not None:
        cv_driver_settings.num_folds = num_folds
    if epochs_per_fold is not None:
        cv_driver_settings.epochs_per_fold = epochs_per_fold

    # build paths object for CrossValidatorDriver
    # config_paths = config_reader.read_path("model.cv_driver")
    # cv_driver_paths = cvd.CrossValidatorDriverPaths(**config_paths)
    
    cv_driver_paths = cvd.CrossValidatorDriverPaths.from_config()

    cv_driver = cvd.CrossValidatorDriver(
        device=gh.get_device(),
        dataset=xmd.X19MGeneralDataset.from_feature_finalizer_output(),
        hyperparameters=hyperparameters,
        settings=cv_driver_settings,
        paths=cv_driver_paths,
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
