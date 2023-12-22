import argparse
import sys
from pathlib import Path

import optuna
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_settings as cfg_settings
import lstm_adversarial_attack.model.model_tuner_driver as td


def main(num_trials: int = None) -> optuna.Study:
    """
    Runs a new optuna study consisting of multiple trials to find
    optimized hyperparameters for use when generating a model  with a
    X19LSTMBuilder and training it with a StandardModelTrainer. Results will be
    saved in a newly created director under
    data/model/hyperparameter_tuning/. If overall study is killed early,
    data from completed trials is still saved.

    :param num_trials: max number of trials to run
    :return: an optuna.Study object with results of completed trials.
    """
    if num_trials is None:
        num_trials = cfg_settings.TUNER_NUM_TRIALS

    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    tuner_driver = td.ModelTunerDriver(
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
