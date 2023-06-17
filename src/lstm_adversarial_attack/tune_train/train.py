import sys
import torch
from pathlib import Path
from torch.utils.data import random_split

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as lcp
import lstm_adversarial_attack.tune_train.trainer_driver as td
import lstm_adversarial_attack.tune_train.tuner_helpers as tuh
import lstm_adversarial_attack.x19_mort_general_dataset as xmd


if __name__ == "__main__":
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    dataset = xmd.X19MGeneralDataset.from_feature_finalizer_output()
    train_dataset_fraction = 0.8
    train_dataset_size = int(len(dataset) * train_dataset_fraction)
    test_dataset_size = len(dataset) - train_dataset_size
    train_dataset, test_dataset = random_split(
        dataset=dataset, lengths=(train_dataset_size, test_dataset_size)
    )
    train_eval_pair = tuh.TrainEvalDatasetPair(
        train=train_dataset, validation=test_dataset
    )

    # driver = td.TrainerDriver.from_optuna_study_path(
    #     train_device=cur_device,
    #     eval_device=cur_device,
    #     train_eval_dataset_pair=train_eval_pair,
    #     study_path=lcp.ONGOING_TUNING_STUDY_PICKLE,
    # )

    #  When using either of two blocks below, get big jump in loss
    #  on restart if first round of training had many (e.g. 400) epochs.
    #  Believe to be due to random state and/or Adam issue.

    # output_dir = lcp.TRAINING_OUTPUT_DIR / "2023-06-17_15_14_45.375162"
    # driver = td.TrainerDriver.from_previous_training(
    #     train_device=cur_device,
    #     eval_device=cur_device,
    #     checkpoint_file=output_dir
    #     / "checkpoints"
    #     / "2023-06-17_15_26_15.852165.tar",
    #     hyperparameters_file=output_dir / "hyperparameters.pickle",
    #     additional_output_dir=output_dir,
    #     train_eval_dataset_pair=train_eval_pair
    # )

    driver = td.TrainerDriver.from_standard_previous_training(
        train_device=cur_device,
        eval_device=cur_device,
        train_eval_dataset_pair=train_eval_pair,
        training_output_dir=lcp.TRAINING_OUTPUT_DIR
        / "2023-06-17_15_14_45.375162",
    )

    cur_train_eval_pair = driver.run(
        num_epochs=1000,
        eval_interval=10,
        evals_per_checkpoint=1,
        save_checkpoints=True,
    )
