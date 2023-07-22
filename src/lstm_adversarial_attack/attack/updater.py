import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfg_path
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.attack.attack_hyperparameter_tuner as aht


msn_dirname = "2023-06-28_12_11_46.874267"
ms_dirname = "2023-06-30_10_52_24.059982"
mss_dirname = "2023-07-01_01_39_18.441890"
mssm_dirname = "2023-07-01_11_03_13.591090"


aht_dirs = [msn_dirname, ms_dirname, mss_dirname, mssm_dirname]
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



# for driver in tuner_drivers:
#     driver.sample_selection_seed = 13579
#
# for idx, driver in enumerate(tuner_drivers):
#     rio.ResourceExporter().export(
#         resource=driver.__dict__,
#         path=attack_tuner_driver_paths[idx].parent
#         / "attack_tuner_driver_dict.pickle",
#     )

# objective_extra_kwargs = [
#     {"max_num_perts": 1},
#     {},
#     {},
#     {}
# ]
#
# for idx, driver in enumerate(tuner_drivers):
#     driver.objective_extra_kwargs = objective_extra_kwargs[idx]
#
#
# for idx, driver in enumerate(tuner_drivers):
#     rio.ResourceExporter().export(resource=driver, path=attack_tuner_driver_paths[idx])
