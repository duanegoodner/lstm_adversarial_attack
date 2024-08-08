import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.model.model_data_structs as mds
import lstm_adversarial_attack.utils.gpu_helpers as gh
import lstm_adversarial_attack.model.cross_validator_driver as cvd
import lstm_adversarial_attack.utils.path_searches as ps
import lstm_adversarial_attack.utils.session_id_generator as sig
from lstm_adversarial_attack.config.read_write import PATH_CONFIG_READER


def main(
    redirect_terminal_output: bool, model_tuning_id: str = None
) -> dict[int, mds.TrainEvalLogPair]:
    cv_training_id = sig.generate_session_id()

    model_tuning_output_root = Path(
        PATH_CONFIG_READER.read_path("model.tuner_driver.output_dir")
    )

    model_training_output_root = Path(
        PATH_CONFIG_READER.read_path("model.cv_driver.output_dir")
    )

    if model_tuning_id is None:
        model_tuning_id = ps.get_latest_sequential_child_dirname(
            root_dir=model_tuning_output_root
        )

    print(
        f"Starting new cross-validation training session {cv_training_id}.\n"
        f"Using model hyperparameters from model tuning session "
        f"{model_tuning_id}.\n\n"
        
        f"To monitor training in tensorboard, run the following command in "
        f"another terminal:\n"
        f"tensorboard --logdir {model_training_output_root}/{cv_training_id}/"
        f"tensorboard --host=0.0.0.0\n"
        f"Then go to http://localhost:6006/ in your browser."
    )

    cv_driver = cvd.CrossValidatorDriver(
        model_tuning_id=model_tuning_id,
        cv_training_id=cv_training_id,
        device=gh.get_device(),
        redirect_terminal_output=redirect_terminal_output,
    )

    return cv_driver.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs cross-validation training using the best model "
        "hyperparameters from a particular model tuning session."
    )
    parser.add_argument(
        "-t",
        "--model_tuning_id",
        type=str,
        action="store",
        nargs="?",
        help="ID of model tuning session that hyperparameters are obtained "
             "from. Defaults to the most recently created session.",
    )
    parser.add_argument(
        "-r",
        "--redirect",
        action="store_true",
        help="Redirect terminal output to log file",
    )

    args_namespace = parser.parse_args()

    all_fold_results = main(
        redirect_terminal_output=args_namespace.redirect,
        model_tuning_id=args_namespace.model_tuning_id,
    )
