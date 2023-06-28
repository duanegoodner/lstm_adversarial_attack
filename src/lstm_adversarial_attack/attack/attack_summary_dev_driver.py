import lstm_adversarial_attack.attack.attack_summary as asu
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.resource_io as rio


# result_path = (
#     cfg_paths.FROZEN_HYPERPARAMETER_ATTACK
#     / "2023-06-24_16_30_10.505396"
#     / "2023-06-24_18_31_50.279321_final_attack_result.pickle"
# )

result_path = (
    cfg_paths.ATTACK_OUTPUT_DIR
    / "2023-06-23_19_49_49.918382.pickle"
)

trainer_result = rio.ResourceImporter().import_pickle_to_object(
    path=result_path
)

attack_summary = asu.AttackResults(trainer_result=trainer_result)


# thing = attack_summary.best_perts_summary