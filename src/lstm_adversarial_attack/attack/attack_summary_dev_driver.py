import lstm_adversarial_attack.attack.attack_summary as asu
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.attack.attack_results_analyzer as ara

# build attack summary
result_path = (
    cfg_paths.FROZEN_HYPERPARAMETER_ATTACK
    / "2023-06-28_17_50_46.701620"
    / "2023-06-28_19_02_16.499026_final_attack_result.pickle"
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

susceptibility_calculator = ara.SusceptibilityCalculator(
    attack_summary=attack_summary
)


