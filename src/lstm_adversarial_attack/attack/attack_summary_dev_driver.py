import lstm_adversarial_attack.attack.attack_summary as asu
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.resource_io as rio


result_path = (
    cfg_paths.FROZEN_HYPERPARAMETER_ATTACK
    / "2023-06-28_17_50_46.701620"
    / "2023-06-28_19_02_16.499026_final_attack_result.pickle"
)

# result_path = (
#     cfg_paths.ATTACK_OUTPUT_DIR
#     / "2023-06-23_19_49_49.918382.pickle"
# )

trainer_result = rio.ResourceImporter().import_pickle_to_object(
    path=result_path
)

attack_summary = asu.AttackResults(trainer_result=trainer_result)


thing = attack_summary.best_examples_df