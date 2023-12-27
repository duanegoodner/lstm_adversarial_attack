import sys
from datetime import datetime

import torch
from pathlib import Path
import sklearn.model_selection
from torch.utils.data import Dataset, Subset, random_split
from typing import Callable

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.model.trainer_driver as td
import lstm_adversarial_attack.model.tuner_helpers as tuh


class CrossValidator:
    """
    Runs K-fold cross-validation
    """

    def __init__(
            self,
            device: torch.device,
            dataset: Dataset,
            hyperparameter_settings: tuh.X19LSTMHyperParameterSettings,
            num_folds: int,
            epochs_per_fold: int,
            eval_interval: int,
            collate_fn: Callable,
            fold_class: sklearn.model_selection.BaseCrossValidator,
            kfold_random_seed: int,
            single_fold_eval_fraction: float,
            cv_output_root_dir: str,
            tuning_study_name: str = None,
    ):
        self.device = device
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.hyperparameter_settings = hyperparameter_settings
        self.num_folds = num_folds
        self.epochs_per_fold = epochs_per_fold
        self.eval_interval = eval_interval
        self.fold_class = fold_class
        self.kfold_random_seed = kfold_random_seed
        self.cv_datasets = self.create_datasets()
        self.single_fold_eval_fraction = single_fold_eval_fraction
        self.cv_output_root_dir = cv_output_root_dir
        self.output_dir = self.create_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tuning_study_name = tuning_study_name

    def create_output_dir(self):
        timestamp = "".join(
            char for char in str(datetime.now()) if char.isdigit()
        )
        return Path(self.cv_output_root_dir) / f"cv_training_{timestamp}"

    def create_single_fold_dataset(
            self
    ) -> list[tuh.TrainEvalDatasetPair]:
        train_dataset_size = int(
            len(self.dataset) * (1 - self.single_fold_eval_fraction))
        test_dataset_size = len(self.dataset) - train_dataset_size
        train_dataset, test_dataset = random_split(
            dataset=self.dataset,
            lengths=(train_dataset_size, test_dataset_size),
        )
        train_eval_pair = tuh.TrainEvalDatasetPair(
            train=train_dataset, validation=test_dataset
        )
        return [train_eval_pair]

    def create_multi_fold_datasets(self) -> list[tuh.TrainEvalDatasetPair]:
        """
        Generates train/eval datasets (subsets of full dataset) for each fold
        :return: K pairs of train/eval datasets
        """
        fold_generator_builder = self.fold_class(
            n_splits=self.num_folds,
            shuffle=True,
            random_state=self.kfold_random_seed,
        )
        fold_generator = fold_generator_builder.split(
            self.dataset[:][0], self.dataset[:][1]
        )

        all_train_eval_pairs = []

        for fold_idx, (train_indices, validation_indices) in enumerate(
                fold_generator
        ):
            train_dataset = Subset(dataset=self.dataset, indices=train_indices)
            validation_dataset = Subset(
                dataset=self.dataset, indices=validation_indices
            )
            all_train_eval_pairs.append(
                tuh.TrainEvalDatasetPair(
                    train=train_dataset, validation=validation_dataset
                )
            )

        return all_train_eval_pairs

    def create_datasets(self) -> list[tuh.TrainEvalDatasetPair]:
        """
        Generates train/eval datasets (subsets of full dataset) for each fold
        :return: K pairs of train/eval datasets
        """
        if self.num_folds == 1:
            return self.create_single_fold_dataset()
        else:
            return self.create_multi_fold_datasets()

    def run_fold(
            self, fold_idx: int, train_eval_pair: tuh.TrainEvalDatasetPair
    ):
        """
        Runs train/eval sequences for a single fold.

        Results get saved under self.root_output_dir
        :param fold_idx: index of fold
        :param train_eval_pair: train/eval dataset pair for current fold
        """
        trainer_driver = td.TrainerDriver(
            device=self.device,
            hyperparameter_settings=self.hyperparameter_settings,
            model=tuh.X19LSTMBuilder(
                settings=self.hyperparameter_settings
            ).build(),
            train_eval_dataset_pair=train_eval_pair,
            fold_idx=fold_idx,
            output_dir=self.output_dir,
            summary_writer_subgroup=f"fold_{fold_idx}",
            summary_writer_add_graph=fold_idx == 0,
        )

        trainer_driver.run(
            num_epochs=self.epochs_per_fold,
            eval_interval=self.eval_interval,
            save_checkpoints=True,
        )

    def run_all_folds(self):
        """
        Runs train/eval sequence on all folds.

        Results saved under self.root_output_dir
        """

        for fold_idx, train_eval_pair in enumerate(self.cv_datasets):
            self.run_fold(fold_idx=fold_idx, train_eval_pair=train_eval_pair)
