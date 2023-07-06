"""
Dev script for viewing results of attack on model with fixed hyperparams
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.attack.attack_result_data_structs as ads
import lstm_adversarial_attack.config_paths as cfg_paths
# import lstm_adversarial_attack.attack.attack_results_analyzer as ara


result_path = (
    cfg_paths.FROZEN_HYPERPARAMETER_ATTACK
    / "2023-06-28_17_50_46.701620"
    / "2023-06-28_19_02_16.499026_final_attack_result.pickle"
)

trainer_result = rio.ResourceImporter().import_pickle_to_object(path=result_path)
success_summary = ads.TrainerSuccessSummary(trainer_result=trainer_result)

# best_48hr_orig0_maxperts1_summary = success_summary.filtered_examples_summary(
#     recorded_example_type=ads.RecordedExampleType.BEST,
#     seq_length_min=48,
#     seq_length_max=48,
#     orig_label=0,
#     min_num_nonzero_perts=1,
#     max_num_nonzero_perts=1
# )
#
# best_48hr_orig0_maxperts1_analysis = ara.AttackSusceptibilityMetrics(
#     perts=best_48hr_orig0_maxperts1_summary.perts.data
# )
#
# best_48hr_orig1_maxperts1_summary = success_summary.filtered_examples_summary(
#     recorded_example_type=ads.RecordedExampleType.BEST,
#     seq_length_min=48,
#     seq_length_max=48,
#     orig_label=1,
#     min_num_nonzero_perts=1,
#     max_num_nonzero_perts=1
# )
#
# best_48hr_orig1_maxperts1_analysis = ara.AttackSusceptibilityMetrics(
#     perts=best_48hr_orig1_maxperts1_summary.perts.data
# )