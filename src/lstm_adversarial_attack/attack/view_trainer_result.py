import lstm_adversarial_attack.resource_io as rio
from attack_result_data_structs import TrainerSuccessSummary
from lstm_adversarial_attack.config_paths import ATTACK_OUTPUT_DIR

trainer_result = rio.ResourceImporter().import_pickle_to_object(
    path=ATTACK_OUTPUT_DIR
    / "2023-06-06_10:37:34.881361"
    / "train_result.pickle"
)

success_summary = TrainerSuccessSummary(trainer_result=trainer_result)
