import argparse
import sys
from pathlib import Path

import optuna

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.utils.gpu_helpers as gh
import lstm_adversarial_attack.model.model_tuner_driver as td
import lstm_adversarial_attack.utils.path_searches as ps
from lstm_adversarial_attack.config import CONFIG_READER


def main(model_tuning_id: str = None) -> optuna.Study:
    """
    Resumes hyperparameter tuning using the results of a previously run study.
    Results are appended to the previous study results. (Does not create a new
    study)
    :model_tuning_id: ID of model tuning session to resume
    :return: the updated optuna Study.
    """
    cur_device = gh.get_device()

    tuning_output_root = Path(
        CONFIG_READER.read_path("model.tuner_driver.output_dir")
    )

    # If no model_tuning_id provided, use id of latest model tuning
    if model_tuning_id is None:
        model_tuning_id = ps.get_latest_sequential_child_dirname(
            root_dir=tuning_output_root
        )

    print(
        f"Continuing existing model hyperparameter tuning session {model_tuning_id}"
    )

    tuner_driver = td.ModelTunerDriver.from_model_tuning_id(
        device=cur_device,
        model_tuning_id=model_tuning_id,
    )

    study = tuner_driver(
        num_trials=CONFIG_READER.get_config_value(
            "model.tuner_driver.num_trials"
        )
    )

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
