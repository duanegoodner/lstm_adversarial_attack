import argparse
import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_settings as cfg_settings
import lstm_adversarial_attack.tune_train.tuner_driver as td


def main(num_trials: int = None):
    if num_trials is None:
        num_trials = cfg_settings.TUNER_NUM_TRIALS

    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    tuner_driver = td.TunerDriver(
        device=cur_device,
    )

    study = tuner_driver(num_trials=num_trials)

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--num_trials",
        type=int,
        action="store",
        nargs="?",
        help="Number of tuning trials to run. If not provided, default is "
             "value stored in config_setting.TUNER_NUM_TRIALS"
    )
    args_namespace = parser.parse_args()
    completed_study = main(**args_namespace.__dict__)


