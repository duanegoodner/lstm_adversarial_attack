import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.tune_train.tuner_driver as td


if __name__ == "__main__":
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    tuner_driver = td.TunerDriver(
        device=cur_device,
        continue_study_path=cfg_paths.ONGOING_TUNING_STUDY_PICKLE,
        output_dir=cfg_paths.ONGOING_TUNING_STUDY_DIR,
    )
    my_completed_study = tuner_driver(num_trials=30)
