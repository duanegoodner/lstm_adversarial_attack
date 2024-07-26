import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.attack_driver as ad
import lstm_adversarial_attack.attack.attack_result_data_structs as ards
import lstm_adversarial_attack.gpu_helpers as gh
import lstm_adversarial_attack.path_searches as ps
import lstm_adversarial_attack.x19_mort_general_dataset as xmd
from lstm_adversarial_attack.config import CONFIG_READER


def main(attack_tuning_id: str) -> ards.TrainerSuccessSummary:
    """
    Runs attack on dataset
    :param attack_tuning_id: ID of attack tuning session to use for attack hyperparameter selection
    """
    attack_id = "".join(
        char for char in str(datetime.now()) if char.isdigit()
    )

    cur_device = gh.get_device()

    attack_tuning_output_root = Path(
        CONFIG_READER.read_path("attack.tune.output_dir")
    )
    if attack_tuning_id is None:
        attack_tuning_id = ps.get_latest_sequential_child_dirname(
            root_dir=attack_tuning_output_root
        )

    attack_driver = ad.AttackDriver.from_attack_tuning_id(
        attack_tuning_id=attack_tuning_id,
        attack_id=attack_id,
        device=cur_device,
    )

    trainer_result = attack_driver()

    trainer_result_dto = ards.TrainerResultDTO(
        dataset_info=xmd.X19MGeneralDatasetInfo(
            preprocess_id=attack_driver.preprocess_id,
            max_num_samples=attack_driver.max_num_samples,
            random_seed=attack_driver.sample_selection_seed,
        ),
        dataset_indices=trainer_result.dataset_indices,
        epochs_run=trainer_result.epochs_run,
        input_seq_lengths=trainer_result.input_seq_lengths,
        first_examples=trainer_result.first_examples,
        best_examples=trainer_result.best_examples,
    )

    success_summary = ards.TrainerSuccessSummary(trainer_result=trainer_result)

    return success_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--attack_tuning_id",
        action="store",
        nargs="?",
        help=(
            "ID of attack tuning session to use for attack hyperparameter selection"
        ),
    )
    args_namespace = parser.parse_args()
    cur_success_summary = main(**args_namespace.__dict__)
