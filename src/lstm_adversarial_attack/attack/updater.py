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
objective_names = [
    "max_num_nonzero_perts",
    "sparsity",
    "sparse_small",
    "sparse_small_max",
]
attack_tuner_driver_paths = [
    (
        cfg_path.ATTACK_HYPERPARAMETER_TUNING
        / dirname
        / "attack_tuner_driver.pickle"
    )
    for dirname in aht_dirs
]

attack_tuner_driver_dict_paths = [
    (
        cfg_path.ATTACK_HYPERPARAMETER_TUNING
        / dirname
        / "attack_tuner_driver_dict.pickle"
    )
    for dirname in aht_dirs
]



tuner_drivers = [
    rio.ResourceImporter().import_pickle_to_object(path=driver_path)
    for driver_path in attack_tuner_driver_paths
]

tuner_driver_dicts =  [
    rio.ResourceImporter().import_pickle_to_object(path=driver_dict_path)
    for driver_dict_path in attack_tuner_driver_dict_paths
]

for idx, driver_dict in enumerate(tuner_driver_dicts):
    del driver_dict["data_provenance"]
    rio.ResourceExporter().export(
        resource=driver_dict,
        path=attack_tuner_driver_paths[idx].parent
        / "attack_tuner_driver_dict.pickle",
    )


# for idx, driver_dict in enumerate(tuner_driver_dicts):
#     driver_dict["provenance"]["fold"] = 0
#     rio.ResourceExporter().export(
#         resource=driver_dict,
#         path=attack_tuner_driver_paths[idx].parent
#         / "attack_tuner_driver_dict.pickle",
#     )

# for idx, driver in enumerate(tuner_drivers):
#     driver.objective_name = objective_names[idx]
#     delattr(driver, "objective")
#     rio.ResourceExporter().export(
#         resource=driver.__dict__,
#         path=attack_tuner_driver_paths[idx].parent
#         / "attack_tuner_driver_dict.pickle",
#     )
#     rio.ResourceExporter().export(
#         resource=driver,
#         path=attack_tuner_driver_paths[idx].parent
#         / "attack_tuner_driver.pickle",
#     )


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
