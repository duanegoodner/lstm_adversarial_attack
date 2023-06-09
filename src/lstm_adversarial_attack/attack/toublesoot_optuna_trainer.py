import lstm_adversarial_attack.resource_io as rio
from attack_result_data_structs import TrainerSuccessSummary
from lstm_adversarial_attack.config_paths import (
    ATTACK_OUTPUT_DIR,
    ATTACK_HYPERPARAMETER_TUNING,
)
from pathlib import Path

study_result_path = ATTACK_HYPERPARAMETER_TUNING / "2023-06-07_02:10:11.684269"

result_pickles = list(study_result_path.glob("*/*.pickle"))

all_results = [
    rio.ResourceImporter().import_pickle_to_object(path=item)
    for item in result_pickles
]

success_summaries = [
    TrainerSuccessSummary(trainer_result=item) for item in all_results[:-1]
]
