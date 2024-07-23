import argparse
import sys
from pathlib import Path

import optuna

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.gpu_helpers as gh
import lstm_adversarial_attack.model.model_tuner_driver as td
import lstm_adversarial_attack.preprocess.encode_decode as edc
import lstm_adversarial_attack.tuning_db.tuning_studies_database as tsd
from lstm_adversarial_attack.config import CONFIG_READER


def main(model_tuning_id: str = None) -> optuna.Study:
    """
    Resumes hyperparameter tuning using the results of a previously run study.
    Results are appended to the previous study results. (Does not create a new
    study)
    :param study_name: Name of study (as saved in RDB) to resume
    :return: the updated optuna Study.
    """
    tuning_output_root = Path(
        CONFIG_READER.read_path("model.tuner_driver.output_dir")
    )

    if model_tuning_id is None:
        latest_tuning_output_dir = max(
            [
                path
                for path in tuning_output_root.iterdir()
                if tuning_output_root.is_dir()
            ]
        )
        model_tuning_id = latest_tuning_output_dir.name

    model_tuning_output_dir = tuning_output_root / model_tuning_id
    tuner_driver_summary_path = model_tuning_output_dir / f"tuner_driver_summary_{model_tuning_id}.json"
    assert tuner_driver_summary_path.exists()

    study_name = f"model_tuning_{model_tuning_id}"
    assert tsd.MODEL_TUNING_DB.is_in_db(study_name=study_name)

    num_trials = CONFIG_READER.get_config_value(
        "model.tuner_driver.num_trials"
    )
    cur_device = gh.get_device()

    tuner_driver_summary = edc.TunerDriverSummaryReader().import_struct(
        path=tuner_driver_summary_path
    )

    tuner_driver = td.ModelTunerDriver(
        device=cur_device,
        settings=td.ModelTunerDriverSettings(**tuner_driver_summary.settings),
        paths=td.ModelTunerDriverPaths(**tuner_driver_summary.paths),
        preprocess_id=tuner_driver_summary.preprocess_id,
        model_tuning_id=model_tuning_id,
    )

    study = tuner_driver(num_trials=num_trials)

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Runs additional hyperparameter tuning trials using "
            "previously saved ModelTunerDriver and optuna.Study as starting"
            "points. New trial results are added to the existing Study."
        )
    )
    parser.add_argument(
        "-t",
        "--model_tuning_id",
        type=str,
        action="store",
        nargs="?",
        help="ID of model tuning session to resume (Defaults to ID of most recently created session",
    )
    args_namespace = parser.parse_args()
    completed_study = main(**args_namespace.__dict__)
