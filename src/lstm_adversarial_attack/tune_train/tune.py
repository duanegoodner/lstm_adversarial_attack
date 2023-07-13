import argparse
import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.tune_train.tuner_driver as td


def main(continue_default_study: bool, custom_study_path: str):
    """
    Runs optuna trials as part of an optuna study. If study results already
    exist at cfg_paths.ONGOING_TUNING_STUDY_PICKLE, adds to that result.
    Creates new study if doesn't exist.
    :return: optuna Study object
    """
    assert not (continue_default_study and custom_study_path)

    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    if continue_default_study:
        continue_study_path = cfg_paths.ONGOING_TUNING_STUDY_PICKLE
    elif custom_study_path is not None:
        continue_study_path = Path(custom_study_path)
    else:
        continue_study_path = None

    tuner_driver = td.TunerDriver(
        device=cur_device,
        continue_study_path=continue_study_path,
        # output_dir=cfg_paths.ONGOING_TUNING_STUDY_DIR,
    )
    my_completed_study = tuner_driver(num_trials=30)

    return my_completed_study


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--continue_default_study",
        action="store_true",
        help=(
            "Continue tuning with existing optuna.Study at path specified by"
            " config_paths.ONGOING_TUNING_STUDY_PICKLE."
        ),
    )
    parser.add_argument(
        "-c",
        "--custom_study_path",
        type=str,
        action="store",
        nargs="?",
        help="Path to an existing optuna.Study to continue running."
    )

    args_namespace = parser.parse_args()
    main(**args_namespace.__dict__)
