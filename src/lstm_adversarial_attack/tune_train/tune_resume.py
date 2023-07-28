import argparse
import sys
from pathlib import Path

import optuna

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.config_settings as cfg_settings
import lstm_adversarial_attack.gpu_helpers as gh
import lstm_adversarial_attack.path_searches as ps
import lstm_adversarial_attack.tune_train.tuner_driver as td


def main(study_dir: Path = None, num_trials: int = None) -> optuna.Study:
    """
    Resumes hyperparameter tuning using the results of a previously run study.
    Results are appended to the previous study results. (Does not create a new
    study)
    :param study_dir: Output directory of the previously run study. If not
    provided, defaults to directory containing the most recently modified
    optuna_study result under directory data/tune_train/hyperparameter_tuning.
    :param num_trials: max number of additional trials to run
    :return: the updated optuna Study.
    """
    if study_dir is None:
        study_dir = ps.latest_modified_file_with_name_condition(
            component_string="optuna_study.pickle",
            root_dir=cfg_paths.HYPERPARAMETER_OUTPUT_DIR,
            comparison_type=ps.StringComparisonType.EXACT_MATCH
        ).parent.parent
    if num_trials is None:
        num_trials = cfg_settings.TUNER_NUM_TRIALS

    cur_device = gh.get_device()
    tuner_driver = td.TunerDriver(
        device=cur_device, continue_tuning_dir=study_dir
    )

    study = tuner_driver(num_trials=num_trials)

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Runs additional hyperparameter tuning trials using "
            "previously saved TunerDriver and optuna.Study as starting"
            "points. New trial results are added to the existing Study."
        )
    )
    parser.add_argument(
        "-s",
        "--study_dir",
        type=str,
        action="store",
        nargs="?",
        help=(
            "Path to the output directory from previously run Study. Directory "
            "structure must  that of directory created when running a "
            "TunerDriver for the first time. "
        ),
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
        )
    )
    args_namespace = parser.parse_args()
    completed_study = main(**args_namespace.__dict__)
