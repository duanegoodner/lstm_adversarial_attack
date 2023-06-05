import lstm_adversarial_attack.resource_io as rio
from attack_result_data_structs import AttackSummary
from lstm_adversarial_attack.config_paths import ATTACK_OUTPUT_DIR

trainer_result = rio.ResourceImporter().import_pickle_to_object(
    path=ATTACK_OUTPUT_DIR / "2023-06-04_23_05_54.389956.pickle"
).latest_result

attack_summary = AttackSummary.from_trainer_result(
    trainer_result=trainer_result
)

