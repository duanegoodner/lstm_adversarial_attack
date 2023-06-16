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

    # driver = td.TrainerDriver.from_optuna_study_path(
    #     train_device=cur_device,
    #     eval_device=cur_device,
    #     study_path=lcp.ONGOING_TUNING_STUDY_PICKLE
    # )

    driver = td.TrainerDriver.from_previous_training(
        train_device=cur_device,
        eval_device=cur_device,
        checkpoint_file=lcp.TRAINING_OUTPUT_DIR
        / "2023-06-15_18_31_44.691855"
        / "checkpoints"
        / "2023-06-15_18_34_09.987363.tar",
        hyperparameters_file=lcp.TRAINING_OUTPUT_DIR
        / "2023-06-15_18_31_44.691855"
        / "hyperparameters.pickle",
        additional_output_dir=lcp.TRAINING_OUTPUT_DIR
        / "2023-06-15_18_31_44.691855"
    )

    cur_train_eval_pair = driver.run(
        num_cycles=25, epochs_per_cycle=1, save_checkpoints=True
    )
