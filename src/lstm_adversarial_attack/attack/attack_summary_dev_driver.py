import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize

import lstm_adversarial_attack.attack.attack_summary as asu
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.attack.attack_results_analyzer as ara

# build attack summary
# result_path = (
#     cfg_paths.FROZEN_HYPERPARAMETER_ATTACK
#     / "2023-06-28_17_50_46.701620"
#     / "2023-06-28_19_02_16.499026_final_attack_result.pickle"
# )


result_path = (
    cfg_paths.FROZEN_HYPERPARAMETER_ATTACK
    / "2023-06-24_14_28_33.535885"
    / "2023-06-24_14_29_30.331752.pickle"
)


trainer_result = rio.ResourceImporter().import_pickle_to_object(
    path=result_path
)
attack_summary = asu.AttackResults(trainer_result=trainer_result)


# get measurement names
measurement_names_path = (
    cfg_paths.PREPROCESS_OUTPUT_DIR / "measurement_col_names.pickle"
)
measurement_names = rio.ResourceImporter().import_pickle_to_object(
    path=measurement_names_path
)

attack_result_analyzer = ara.AttackResultAnalyzer(
    attack_summary=attack_summary
)

zto_48hr_all_pert_counts_first = attack_result_analyzer.get_attack_analysis(
    example_type=asu.RecordedExampleType.FIRST, seq_length=48, orig_label=0
)
zto_48hr_all_pert_counts_best = attack_result_analyzer.get_attack_analysis(
    example_type=asu.RecordedExampleType.BEST, seq_length=48, orig_label=0
)
