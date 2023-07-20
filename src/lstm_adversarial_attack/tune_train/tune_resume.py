import argparse
import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.config_settings as cfg_settings
import lstm_adversarial_attack.gpu_helpers as gh
import lstm_adversarial_attack.path_searches as ps
import lstm_adversarial_attack.tune_train.tuner_driver as td


def main(study_dir: Path = None, num_trials: int = None):
    if study_dir is None:
        study_dir = ps.most_recently_modified_subdir(
            root_path=cfg_paths.HYPERPARAMETER_OUTPUT_DIR
        )
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
