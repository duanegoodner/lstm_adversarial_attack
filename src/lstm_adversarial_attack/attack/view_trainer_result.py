import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.attack.attack_result_data_structs as ads
import lstm_adversarial_attack.config_paths as lcp

optuna_output_dir = (
    lcp.ATTACK_HYPERPARAMETER_TUNING / "2023-06-09_16:20:47.824222"
)
study_path = optuna_output_dir / "optuna_study.pickle"
trial_result_paths = list(optuna_output_dir.glob("*/*.pickle"))

study = rio.ResourceImporter().import_pickle_to_object(path=study_path)

trial_results = [
    rio.ResourceImporter().import_pickle_to_object(item)
    for item in trial_result_paths
]

success_summaries = [
    ads.TrainerSuccessSummary(trainer_result=item) for item in trial_results
]

filtered_best_perts_summary = success_summaries[0].filtered_perts_summary(
    perts_type="best", seq_length=48, orig_label=1
)
