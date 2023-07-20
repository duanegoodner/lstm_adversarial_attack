import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfg_path
import lstm_adversarial_attack.resource_io as rio


msn_dirname = "2023-06-28_12_11_46.874267"
ms_dirname = "2023-06-30_10_52_24.059982"
mss_dirname = "2023-07-01_01_39_18.441890"
mssm_dirname = "2023-07-01_11_03_13.591090"


aht_dirs = [mssm_dirname, ms_dirname, mss_dirname, mssm_dirname]
attack_tuner_driver_paths = [
    (
        cfg_path.ATTACK_HYPERPARAMETER_TUNING
        / dirname
        / "attack_tuner_driver.pickle"
    )
    for dirname in aht_dirs
]

tuner_drivers = [
    rio.ResourceImporter().import_pickle_to_object(path=driver_path)
    for driver_path in attack_tuner_driver_paths
]
#
# for item in tuner_drivers:
#     item.target_model_path = (
#         item.target_model_path.parent.parent.parent.parent
#         / "tune_train"
#         / "cross_validation"
#         / item.target_model_path.parent.name
#         / item.target_model_path.name
#     )

target_model_paths = [item.target_model_path for item in tuner_drivers]

print(all([item.exists() for item in target_model_paths]))

# for idx, driver in enumerate(tuner_drivers):
#     rio.ResourceExporter().export(
#         resource=driver, path=attack_tuner_driver_paths[idx]
#     )
