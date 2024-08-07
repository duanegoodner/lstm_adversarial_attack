import argparse
import sys
from pathlib import Path

import optuna

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.attack_data_structs as ads
import lstm_adversarial_attack.attack.attack_tuner_driver as atd
import lstm_adversarial_attack.utils.gpu_helpers as gh
import lstm_adversarial_attack.utils.path_searches as ps
import lstm_adversarial_attack.utils.session_id_generator as sig
from lstm_adversarial_attack.config.read_write import PATH_CONFIG_READER


def main(redirect: bool, cv_training_id: str = None) -> optuna.Study:
    """
    Takes arguments in format provided by command line interface and uses them
    to call start_new_tuning().
    :param cv_training_id: ID of cross-validation tuning session to use as source of model to attack
    :param redirect: Boolean flag to indicate whether to redirect tuning session to stdout
    """
    device = gh.get_device()
    attack_tuning_id = sig.generate_session_id()

    cv_output_root = Path(
        PATH_CONFIG_READER.read_path("model.cv_driver.output_dir")
    )

    if cv_training_id is None:
        cv_training_id = ps.get_latest_sequential_child_dirname(
            root_dir=cv_output_root
        )

    tuner_driver = atd.AttackTunerDriver(
        cv_training_id=cv_training_id,
        attack_tuning_id=attack_tuning_id,
        attack_tuning_ranges=ads.AttackTuningRanges(),
        settings=ads.AttackTunerDriverSettings.from_config(),
        paths=ads.AttackTunerDriverPaths.from_config(),
        device=device,
        redirect_terminal_output=redirect
    )

    return tuner_driver()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Starts a new study for tuning attack hyperparameters."
    )
    parser.add_argument(
        "-t",
        "--cv_training_id",
        type=str,
        action="store",
        nargs="?",
        help=(
            "ID of cross validation tuning session providing trained model for attack."
        ),
    )
    parser.add_argument(
        "-r",
        "--redirect",
        action="store_true",
        help="Redirect terminal output to log file",
    )

    args_namespace = parser.parse_args()
    completed_study = main(**args_namespace.__dict__)
