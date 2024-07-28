import argparse
import sys
from datetime import datetime
from pathlib import Path

import optuna

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.attack_tuner_driver as atd
import lstm_adversarial_attack.gpu_helpers as gh
import lstm_adversarial_attack.path_searches as ps
from lstm_adversarial_attack.config import CONFIG_READER


def main(
    cv_training_id: str = None,
) -> optuna.Study:
    """
    Takes arguments in format provided by command line interface and uses them
    to call start_new_tuning().
    :param cv_training_id: ID of cross-validation training session to use as source of model to attack
    """
    device = gh.get_device()
    attack_tuning_id = "".join(
        char for char in str(datetime.now()) if char.isdigit()
    )

    cv_output_root = Path(
        CONFIG_READER.read_path("model.cv_driver.output_dir")
    )

    if cv_training_id is None:
        cv_training_id = ps.get_latest_sequential_child_dirname(
            root_dir=cv_output_root
        )

    tuner_driver = atd.AttackTunerDriver(
        cv_training_id=cv_training_id,
        attack_tuning_id=attack_tuning_id,
        device=device,
    )

    return tuner_driver.run()


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
            "ID of cross validation training session providing trained model for attack."
        ),
    )

    args_namespace = parser.parse_args()
    completed_study = main(**args_namespace.__dict__)
