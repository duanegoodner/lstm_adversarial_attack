from pathlib import Path

import lstm_adversarial_attack.attack.attack_result_data_structs as ards
from lstm_adversarial_attack.config.read_write import PATH_CONFIG_READER

if __name__ == "__main__":

    attack_results_root = PATH_CONFIG_READER.read_path(
        "attack.attack_driver.output_dir"
    )
    attack_results_path = (
        Path(attack_results_root)
        / "20240727113324944266"
        / "final_attack_result_20240727113324944266.json"
    )

    imported_results = ards.ATTACK_TRAINER_RESULT_IO.import_to_struct(
        path=attack_results_path
    )

    pass
