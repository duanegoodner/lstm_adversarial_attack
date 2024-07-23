import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.data_structures as ds
import lstm_adversarial_attack.gpu_helpers as gh
import lstm_adversarial_attack.model.cross_validator_driver as cvd
import lstm_adversarial_attack.model.model_data_structs as mds
import lstm_adversarial_attack.model.tuner_helpers as tuh
import lstm_adversarial_attack.path_searches as ps
import lstm_adversarial_attack.preprocess.encode_decode as edc
import lstm_adversarial_attack.tuning_db.tuning_studies_database as tsd
from lstm_adversarial_attack.config import CONFIG_READER


def main(
    # study_name: str = None,
    model_tuning_id: str = None,
) -> dict[int, ds.TrainEvalLogPair]:
    cv_training_id = "".join(
        char for char in str(datetime.now()) if char.isdigit()
    )

    model_tuning_output_root = Path(
        CONFIG_READER.read_path("model.tuner_driver.output_dir")
    )

    # if study_name is None:
    #     study_name = tsd.MODEL_TUNING_DB.get_latest_study().study_name

    if model_tuning_id is None:
        model_tuning_id = ps.get_latest_sequential_child_dirname(
            root_dir=model_tuning_output_root
        )

    study_name = f"model_tuning_{model_tuning_id}"

    model_tuner_driver_summary = edc.TunerDriverSummaryReader().import_struct(
        path=model_tuning_output_root
        / model_tuning_id
        / f"tuner_driver_summary_{model_tuning_id}.json"
    )

    # model_tuning_output_root = CONFIG_READER.read_path(
    #     "model.tuner_driver.output_dir"
    # )
    # model_tuning_output_dir = Path(model_tuning_output_root) / study_name

    # get hyperparameters from database
    hyperparams_dict = tsd.MODEL_TUNING_DB.get_best_params(
        study_name=study_name
    )
    hyperparameters = tuh.X19LSTMHyperParameterSettings(**hyperparams_dict)

    # tuner_driver_summary_files = [
    #     item
    #     for item in list(model_tuning_output_dir.iterdir())
    #     if item.name.startswith("tuner_driver_summary_")
    #     and item.name.endswith(".json")
    # ]
    # assert len(tuner_driver_summary_files) == 1
    #
    # tuner_driver_summary = edc.TunerDriverSummaryReader().import_struct(
    #     path=tuner_driver_summary_files[0]
    # )

    cv_driver_settings = mds.CrossValidatorDriverSettings.from_config()
    cv_driver_paths = mds.CrossValidatorDriverPaths.from_config()

    cv_driver = cvd.CrossValidatorDriver(
        preprocess_id=model_tuner_driver_summary.preprocess_id,
        cv_training_id=cv_training_id,
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
        "-t",
        "--model_tuning_id",
        type=str,
        action="store",
        nargs="?",
        help=(
            "ID of model tuning session to obtain hyperparameters from"
        ),
    )

    args_namespace = parser.parse_args()

    all_fold_results = main(**args_namespace.__dict__)
