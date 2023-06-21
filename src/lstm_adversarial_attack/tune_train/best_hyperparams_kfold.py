import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import lstm_adversarial_attack.tune_train.cross_validator_driver as cvd
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.x19_mort_general_dataset as xmd


def main():
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    cv_driver = cvd.CrossValidatorDriver.from_study_path(
        device=cur_device,
        dataset=xmd.X19MGeneralDataset.from_feature_finalizer_output(),
        study_path=cfg_paths.ONGOING_TUNING_STUDY_PICKLE
    )
    cv_driver.run()


if __name__ == "__main__":
    main()
