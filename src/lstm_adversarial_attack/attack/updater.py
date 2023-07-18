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

tuner_drivers[0].objective = aht.AttackTunerObjectivesBuilder.max_num_nonzero_perts(max_perts=1)
tuner_drivers[1].objective = aht.AttackTunerObjectivesBuilder.sparsity()
tuner_drivers[2].objective = aht.AttackTunerObjectivesBuilder.sparse_small()
tuner_drivers[3].objective = aht.AttackTunerObjectivesBuilder.sparse_small_max()

for idx, driver in enumerate(tuner_drivers):
    rio.ResourceExporter().export(resource=driver, path=attack_tuner_driver_paths[idx])



