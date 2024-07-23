import argparse
import sys
from pathlib import Path



sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.data_structures as ds
import lstm_adversarial_attack.gpu_helpers as gh
import lstm_adversarial_attack.model.cross_validator_driver as cvd
import lstm_adversarial_attack.model.tuner_helpers as tuh
import lstm_adversarial_attack.tuning_db.tuning_studies_database as tsd
import lstm_adversarial_attack.model.model_data_structs as mds
import lstm_adversarial_attack.preprocess.encode_decode as edc
from lstm_adversarial_attack.config import CONFIG_READER


def main(
    study_name: str = None,
) -> dict[int, ds.TrainEvalLogPair]:

    if study_name is None:
        study_name = tsd.MODEL_TUNING_DB.get_latest_study().study_name

    model_tuning_output_root = CONFIG_READER.read_path(
        "model.tuner_driver.output_dir"
    )
    model_tuning_output_dir = Path(model_tuning_output_root) / study_name

    # get hyperparameters from database
    hyperparams_dict = tsd.MODEL_TUNING_DB.get_best_params(
        study_name=study_name
    )
    hyperparameters = tuh.X19LSTMHyperParameterSettings(**hyperparams_dict)

    tuner_driver_summary_files = [
        item
        for item in list(model_tuning_output_dir.iterdir())
        if item.name.startswith("tuner_driver_summary_")
        and item.name.endswith(".json")
    ]
    assert len(tuner_driver_summary_files) == 1

    tuner_driver_summary = edc.TunerDriverSummaryReader().import_struct(
        path=tuner_driver_summary_files[0]
    )

    cv_driver_settings = mds.CrossValidatorDriverSettings.from_config()
    cv_driver_paths = mds.CrossValidatorDriverPaths.from_config()

    cv_driver = cvd.CrossValidatorDriver(
        preprocess_id=tuner_driver_summary.preprocess_id,
        device=gh.get_device(),
        hyperparameters=hyperparameters,
        settings=cv_driver_settings,
        paths=cv_driver_paths,
        tuning_study_name=study_name,
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

    args_namespace = parser.parse_args()

    all_fold_results = main(**args_namespace.__dict__)
