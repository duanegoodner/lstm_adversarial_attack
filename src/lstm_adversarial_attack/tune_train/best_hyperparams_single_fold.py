import sys
import torch
from torch.utils.data import Dataset
from pathlib import Path
from torch.utils.data import random_split

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.config_settings as lcs
import lstm_adversarial_attack.tune_train.trainer_driver as td
import lstm_adversarial_attack.tune_train.tuner_helpers as tuh
import lstm_adversarial_attack.x19_mort_general_dataset as xmd


class SingleFoldTrainer:
    """
    Runs train eval cycle on single fold from dataset
    """
    def __init__(
        self,
        device: torch.device,
        dataset: Dataset,
        train_dataset_fraction: float,
    ):
        """
        :param device: device to run on
        :param dataset: full dataset
        :param train_dataset_fraction: fraction of full dataset for train set
        """
        self.device = device
        self.dataset = dataset
        self.train_dataset_fraction = train_dataset_fraction

    def run(self):
        """
        splits dataset into test/train, then instantiates & runs TrainerDriver
        :return:
        :rtype:
        """
        train_dataset_size = int(
            len(self.dataset) * self.train_dataset_fraction
        )
        test_dataset_size = len(self.dataset) - train_dataset_size
        train_dataset, test_dataset = random_split(
            dataset=self.dataset,
            lengths=(train_dataset_size, test_dataset_size),
        )
        train_eval_pair = tuh.TrainEvalDatasetPair(
            train=train_dataset, validation=test_dataset
        )

        driver = td.TrainerDriver.from_optuna_study_path(
            device=self.device,
            train_eval_dataset_pair=train_eval_pair,
            study_path=cfg_paths.ONGOING_TUNING_STUDY_PICKLE,
        )

        #  When trying to restart post-checkpoint w/ code below instead, get big jump in
        #  loss on restart if first round of training had many (e.g. 400) epochs.
        #  Believe to be Adam issue. Others have reported but no good fix.
        # driver = td.TrainerDriver.from_standard_previous_training(
        #     train_device=cur_device,
        #     eval_device=cur_device,
        #     train_eval_dataset_pair=train_eval_pair,
        #     training_output_dir=cfg_paths.SINGLE_FOLD_OUTPUT_DIR
        #     / "2023-06-17_18_06_39.996609",
        # )


        driver.run(
            num_epochs=lcs.CV_DRIVER_EPOCHS_PER_FOLD,
            eval_interval=lcs.CV_DRIVER_EVAL_INTERVAL,
            evals_per_checkpoint=lcs.CV_DRIVER_EVALS_PER_CHECKPOINT,
            save_checkpoints=True,
        )


def main():
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    trainer = SingleFoldTrainer(
        device=cur_device,
        dataset=xmd.X19MGeneralDataset.from_feature_finalizer_output(),
        train_dataset_fraction=1 - 1 / lcs.CV_DRIVER_NUM_FOLDS,
    )

    trainer.run()


if __name__ == "__main__":
    main()
