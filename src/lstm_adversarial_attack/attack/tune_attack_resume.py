import argparse
import sys
from pathlib import Path

import optuna

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.attack_tuner_driver as atd
import lstm_adversarial_attack.gpu_helpers as gh
import lstm_adversarial_attack.path_searches as ps
from lstm_adversarial_attack.config import CONFIG_READER


def main(
    attack_tuning_id: str = None,
) -> optuna.Study:
    """
    Tunes hyperparameters of an AdversarialAttackTrainer and its
    AdversarialAttacker. Can accept target_model_dir OR existing_study_dir or
    neither, but not both. If no args provided, starts new study using most
    recent cross-validation training results to build target model.

    :param attack_tuning_id: ID of attack tuning session to resume
    :return: an optuna.Study object with results of completed trials
    """

    attack_tuning_output_root = Path(
        CONFIG_READER.read_path("attack.tune.output_dir")
    )

    if attack_tuning_id is None:
        attack_tuning_id = ps.get_latest_sequential_child_dirname(
            root_dir=attack_tuning_output_root
        )

    attack_tuner_driver = atd.AttackTunerDriver.from_attack_tuning_id(
        attack_tuning_id=attack_tuning_id,
        device=gh.get_device(),
    )

    optuna.logging.set_verbosity(optuna.logging.INFO)
    continued_study = attack_tuner_driver.run()
    return continued_study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Runs hyperparameter tuning on the attack algorithm. "
            "If no args passed, will start a new Optuna study "
            "using the model data from the most recent data saved in "
            "data/model/cross_validation (uses fold with "
            "median or near median best performance)"
        )
    )
    parser.add_argument(
        "-t",
        "--attack_tuning_id",
        type=str,
        action="store",
        nargs="?",
        help="ID of attack tuning session to resume",
    )

    args_namespace = parser.parse_args()
    main(**args_namespace.__dict__)
