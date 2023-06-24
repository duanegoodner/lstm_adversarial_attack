import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.attack.attack_result_data_structs as ads
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.attack.attack_results_analyzer as ara


result_path = (
    cfg_paths.FROZEN_HYPERPARAMETER_ATTACK
    / "2023-06-24_16_30_10.505396"
    / "2023-06-24_18_31_50.279321_final_attack_result.pickle"
)

trainer_result = rio.ResourceImporter().import_pickle_to_object(path=result_path)
success_summary = ads.TrainerSuccessSummary(trainer_result=trainer_result)