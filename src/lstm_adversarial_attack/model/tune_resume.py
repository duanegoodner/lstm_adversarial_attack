import argparse
import sys
from pathlib import Path

import optuna

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config as config
import lstm_adversarial_attack.gpu_helpers as gh
import lstm_adversarial_attack.model.model_tuner_driver as td
import lstm_adversarial_attack.path_searches as ps
import lstm_adversarial_attack.preprocess.encode_decode as edc
import lstm_adversarial_attack.tuning_db.tuning_studies_database as tsd


def main(study_name: str = None, num_trials: int = None) -> optuna.Study:
    """
    Resumes hyperparameter tuning using the results of a previously run study.
    Results are appended to the previous study results. (Does not create a new
    study)
    :param study_name: Name of study (as saved in RDB) to resume
    :param num_trials: max number of additional trials to run
    :return: the updated optuna Study.
    """
    if study_name is None:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study_name = tsd.MODEL_TUNING_DB.get_latest_study().study_name
        optuna.logging.set_verbosity(optuna.logging.INFO)

    config_reader = config.ConfigReader()
    tuning_output_dir = config_reader.read_path(
        "model.tuner_driver.tuning_output_dir")

    study_dir = Path(tuning_output_dir) / study_name
    if num_trials is None:
        num_trials = config_reader.get_config_value(
            "model.tuner_driver.num_trials")
    cur_device = gh.get_device()

    tuner_driver_summary_path = ps.latest_modified_file_with_name_condition(
        component_string="tuner_driver_summary_",
        root_dir=study_dir,
        comparison_type=ps.StringComparisonType.PREFIX,
    )
    tuner_driver_summary = edc.TunerDriverSummaryReader().import_struct(
        path=tuner_driver_summary_path
    )

    tuner_driver = td.ModelTunerDriver(
        device=cur_device,
        settings=td.ModelTunerDriverSettings(**tuner_driver_summary.settings),
        paths=td.ModelTunerDriverPaths(**tuner_driver_summary.paths),
        study_name=study_name
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
        "-s",
        "--study_name",
        type=str,
        action="store",
        nargs="?",
        help="Name (as saved in RDB) of study to resume",
    )
    parser.add_argument(
        "-n",
        "--num_trials",
        type=int,
        action="store",
        nargs="?",
        help=(
            "Number of additional trials to run for the study in study_dir. "
            "Defaults to value stored in config_settings.TUNER_NUM_TRIALS."
        ),
    )
    args_namespace = parser.parse_args()
    completed_study = main(**args_namespace.__dict__)
