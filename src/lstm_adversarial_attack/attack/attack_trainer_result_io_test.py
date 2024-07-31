import lstm_adversarial_attack.attack.attack_result_data_structs as ards
from pathlib import Path
import lstm_adversarial_attack.config_paths as cfg_paths


if __name__ == "__main__":

    attack_result_path = (
        cfg_paths.FROZEN_HYPERPARAMETER_ATTACK
        / "20240727113324944266"
        / "final_attack_result_20240727113324944266.json"
    )

    imported_results = ards.ATTACK_TRAINER_RESULT_IO.import_to_struct(
        path=attack_result_path
    )

    pass