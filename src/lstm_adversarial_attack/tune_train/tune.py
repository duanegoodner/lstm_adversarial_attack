import argparse
import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.tune_train.model_tuner_driver as td


def main(continue_default_tuning: bool, custom_study_path: str):
    """
    Runs optuna trials as part of an optuna study. If study results already
    exist at cfg_paths.ONGOING_TUNING_STUDY_PICKLE, adds to that result.
    Creates new study if doesn't exist.
    :return: optuna Study object
    """
    assert not (continue_default_tuning and custom_study_path)

    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    if continue_default_tuning:
        continue_tuning_dir = cfg_paths.ONGOING_TUNING_STUDY_DIR
    elif custom_study_path is not None:
        continue_tuning_dir = Path(custom_study_path)
    else:
        continue_tuning_dir = None

    tuner_driver = td.ModelTunerDriver(
        device=cur_device,
        continue_tuning_dir=continue_tuning_dir,
        # output_dir=cfg_paths.ONGOING_TUNING_STUDY_DIR,
    )
    my_completed_study = tuner_driver(num_trials=30)

    return my_completed_study


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--continue_default_tuning",
        action="store_true",
        help=(
            "Continue prior tuning that has existing output in directory:"
            "config_paths.ONGOING_TUNING_STUDY_DIR."
        ),
    )
    parser.add_argument(
        "-c",
        "--custom_study_path",
        type=str,
        action="store",
        nargs="?",
        help=(
            "Path to the output directory of an existing tuning study. "
            "Note: This directory must match output file structure of a"
            " HyperParameterTuner.output_dir."
            " custom_study_path/checkpoints_tuner/optuna_study.pickle must"
            " exist"
        ),
    )

    args_namespace = parser.parse_args()
    main(**args_namespace.__dict__)
