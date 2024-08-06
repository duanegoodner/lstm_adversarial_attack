import argparse
import sys
from pathlib import Path

import optuna

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.model.model_tuner_driver as td
import lstm_adversarial_attack.utils.gpu_helpers as gh
import lstm_adversarial_attack.utils.path_searches as ps
from lstm_adversarial_attack.config.read_write import CONFIG_READER, PATH_CONFIG_READER


def main(redirect: bool, model_tuning_id: str = None) -> optuna.Study:
    """
    Resumes hyperparameter tuning using the results of a previously run study.
    Results are appended to the previous study results. (Does not create a new
    study)
    :model_tuning_id: ID of model tuning session to resume
    :return: the updated optuna Study.
    """
    tuning_output_root = Path(
        PATH_CONFIG_READER.read_path("model.tuner_driver.output_dir")
    )
    # If no model_tuning_id provided, use id of latest model tuning
    if model_tuning_id is None:
        model_tuning_id = ps.get_latest_sequential_child_dirname(
            root_dir=tuning_output_root
        )
    cur_device = gh.get_device()

    print(
        f"Continuing existing model hyperparameter tuning session {model_tuning_id}\n"
        f"To monitor tuning in tensorboard, run the following command in another terminal:\n"
        f"tensorboard --logdir=/home/devspace/project/data/model/cross_validation/"
        f"{model_tuning_id}/tensorboard --host=0.0.0.0' in a different "
        f"terminal\n"
        f"Then go to http://localhost:6006/ in your browser.\n"
    )

    tuner_driver = td.ModelTunerDriver.from_model_tuning_id(
        device=cur_device,
        model_tuning_id=model_tuning_id,
        redirect_terminal_output=redirect
    )

    study = tuner_driver(
        num_trials=CONFIG_READER.get_value(
            "model.tuner_driver.num_trials"
        )
    )

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Runs additional model hyperparameter tuning trials for an existing "
            "tuning study. Uses previously saved ModelTunerDriver and "
            "optuna.Study as starting points. New trial results are added to "
            "the existing Study."
        )
    )
    parser.add_argument(
        "-t",
        "--model_tuning_id",
        type=str,
        action="store",
        nargs="?",
        help="ID of model tuning session to resume. Defaults to ID of most "
             "recently created session.",
    )
    parser.add_argument(
        "-r",
        "--redirect",
        action="store_true",
        help="Redirect terminal output to log file",
    )
    args_namespace = parser.parse_args()
    completed_study = main(
        redirect=args_namespace.redirect,
        model_tuning_id=args_namespace.model_tuning_id,
    )
