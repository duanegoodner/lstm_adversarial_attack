import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.attack.attack_result_data_structs as ads
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.attack.attack_results_analyzer as ara


study_out_dir = (
    cfg_paths.ATTACK_HYPERPARAMETER_TUNING / "2023-06-22_07_32_01.590696"
)

importer = rio.ResourceImporter()

tuner = importer.import_pickle_to_object(
    path=study_out_dir / "attack_hyperparameter_tuner.pickle"
)
driver = importer.import_pickle_to_object(
    path=study_out_dir
)

