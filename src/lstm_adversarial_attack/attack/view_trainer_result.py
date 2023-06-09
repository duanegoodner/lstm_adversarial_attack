import lstm_adversarial_attack.resource_io as rio
from attack_result_data_structs import TrainerSuccessSummary
from lstm_adversarial_attack.config_paths import ATTACK_HYPERPARAMETER_TUNING

optuna_output_dir = ATTACK_HYPERPARAMETER_TUNING / "2023-06-09_16:20:47.824222"
trial_result_paths = list(optuna_output_dir.glob("*/*.pickle"))

trial_results = [
    rio.ResourceImporter().import_pickle_to_object(item)
    for item in trial_result_paths
]

success_summaries = [
    TrainerSuccessSummary(trainer_result=item) for item in trial_results
]
# trainer_result = rio.ResourceImporter().import_pickle_to_object(
#     path=ATTACK_HYPERPARAMETER_TUNING / "2023-06-09_17:05:18.844499"
#     / "2023-06-09_16:21:08.240843"
#     / "train_result.pickle"
# )


# success_summary = TrainerSuccessSummary(trainer_result=trainer_result)
