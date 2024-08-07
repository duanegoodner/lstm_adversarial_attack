import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.attack_driver as ad
import lstm_adversarial_attack.attack.attack_result_data_structs as ards
import lstm_adversarial_attack.utils.gpu_helpers as gh
import lstm_adversarial_attack.utils.path_searches as ps
import lstm_adversarial_attack.utils.session_id_generator as sig
from lstm_adversarial_attack.config.read_write import PATH_CONFIG_READER


def main(attack_tuning_id: str, redirect: bool) -> ards.TrainerSuccessSummary:
    """
    Runs attack on dataset
    :param attack_tuning_id: ID of attack tuning session to use for attack hyperparameter selection
    """
    attack_id = sig.generate_session_id()

    cur_device = gh.get_device()

    attack_tuning_output_root = Path(
        PATH_CONFIG_READER.read_path("attack.tune.output_dir")
    )
    if attack_tuning_id is None:
        attack_tuning_id = ps.get_latest_sequential_child_dirname(
            root_dir=attack_tuning_output_root
        )

    attack_driver = ad.AttackDriver(
        attack_tuning_id=attack_tuning_id,
        attack_id=attack_id,
        device=cur_device,
        redirect_terminal_output=redirect
    )

    print(
        f"Starting new model attack session {attack_id}.\n"
        f"Using attack hyperparameters from model attack tuning session "
        f"{attack_driver.attack_tuning_id}\n"
        f"Using model from model training session {attack_driver.summary.cv_training_id}"
    )

    attack_trainer_result = attack_driver()

    success_summary = ards.TrainerSuccessSummary(
        attack_trainer_result=attack_trainer_result
    )

    return success_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--attack_tuning_id",
        action="store",
        nargs="?",
        help=(
            "ID of attack tuning session to use for attack hyperparameter selection"
        ),
    )
    parser.add_argument(
        "-r",
        "--redirect",
        action="store_true",
        help="Redirect terminal output to log file",
    )
    args_namespace = parser.parse_args()
    cur_success_summary = main(**args_namespace.__dict__)
