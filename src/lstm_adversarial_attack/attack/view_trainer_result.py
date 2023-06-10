import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.attack.attack_result_data_structs as ads
import lstm_adversarial_attack.config_paths as lcp

optuna_output_dir = lcp.ATTACK_HYPERPARAMETER_TUNING / "2023-06-09_16:20:47.824222"
trial_result_paths = list(optuna_output_dir.glob("*/*.pickle"))

trial_results = [
    rio.ResourceImporter().import_pickle_to_object(item)
    for item in trial_result_paths
]

success_summaries = [
    ads.TrainerSuccessSummary(trainer_result=item) for item in trial_results
]
# trainer_result = rio.ResourceImporter().import_pickle_to_object(
#     path=ATTACK_HYPERPARAMETER_TUNING / "2023-06-09_17:05:18.844499"
#     / "2023-06-09_16:21:08.240843"
#     / "train_result.pickle"
# )


# success_summary = TrainerSuccessSummary(trainer_result=trainer_result)
