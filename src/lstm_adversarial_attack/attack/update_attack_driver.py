import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfg_path
import lstm_adversarial_attack.resource_io as rio

full_attack_dir_names = [
    "2023-06-28_17_50_46.701620",
    "2023-06-30_12_05_34.834996",
    "2023-07-01_12_01_25.552909",
]

attack_driver_pickle_paths = [
    cfg_path.FROZEN_HYPERPARAMETER_ATTACK / dir_name / "attack_driver.pickle"
    for dir_name in full_attack_dir_names
]

attack_driver_pickles = [
    rio.ResourceImporter().import_pickle_to_object(path=pickle_path)
    for pickle_path in attack_driver_pickle_paths
]

for idx, driver_pickle in enumerate(attack_driver_pickles):
    rio.ResourceExporter().export(
        resource=driver_pickle.__dict__,
        path=attack_driver_pickle_paths[idx].parent
        / "attack_driver_dict.pickle",
    )
