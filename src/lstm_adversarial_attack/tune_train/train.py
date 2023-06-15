import sys
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as lcp
import lstm_adversarial_attack.tune_train.trainer_driver as td



if __name__ == "__main__":
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    driver = td.TrainerDriver.from_optuna_study_best_trial(
        train_device=cur_device,
        eval_device=cur_device,
        study_path=lcp.ONGOING_TUNING_STUDY_PICKLE
    )

    cur_train_eval_pair = driver.run(
        num_cycles=400, epochs_per_cycle=1, save_checkpoints=True
    )
