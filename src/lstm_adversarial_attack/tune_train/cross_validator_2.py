import optuna
import sys
import torch
import torch.nn as nn
from optuna.pruners import BasePruner, MedianPruner
from optuna.samplers import BaseSampler, TPESampler
from optuna.trial import TrialState
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from typing import Callable

sys.path.append(str(Path(__file__).parent.parent.parent))

import lstm_adversarial_attack.config_settings as lcs
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.data_structures as ds
import lstm_adversarial_attack.tune_train.trainer_driver as td
# import lstm_adversarial_attack.data_structures as ds
import lstm_adversarial_attack.config_paths as lcp
import lstm_adversarial_attack.tune_train.standard_model_trainer as smt
import lstm_adversarial_attack.weighted_dataloader_builder as wdb
import lstm_adversarial_attack.tune_train.tuner_helpers as tuh


class CrossValidator:
    def __init__(
        self,
        device: torch.device,
        dataset: Dataset,
        collate_fn: Callable,
        # model: nn.Module,
        hyperparameter_settings: tuh.X19LSTMHyperParameterSettings,
        num_folds: int,
        epochs_per_fold: int,
        fold_class: Callable = StratifiedKFold,
        kfold_random_seed: int = lcs.TUNER_KFOLD_RANDOM_SEED,
        output_dir: Path = None,
        save_fold_info: bool = True,
    ):
        self.device = device
        self.dataset = dataset
        self.collate_fn = collate_fn
        # self.model = model
        self.hyperparameter_settings = hyperparameter_settings
        self.num_folds = num_folds
        self.epochs_per_fold = epochs_per_fold
        self.fold_class = fold_class
        self.kfold_random_seed = kfold_random_seed
        self.cv_datasets = self.create_datasets()
        self.output_dir = output_dir
        self.save_fold_info = save_fold_info

    def create_datasets(self) -> list[tuh.TrainEvalDatasetPair]:
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

    # def initialize_model(self):
    #     for name, param in self.model.named_parameters():
    #         if "weight" in name:
    #             nn.init.xavier_normal_(param)
    #         elif "bias" in name:
    #             nn.init.constant_(param, 0.0)


    def run_fold(
        self, fold_idx: int, train_eval_pair: tuh.TrainEvalDatasetPair
    ):
        # self.initialize_model()
        trainer_driver = td.TrainerDriver(
            train_device=self.device,
            eval_device=self.device,

        )

    def run_all_folds(self):
        for fold_idx, train_eval_pair in enumerate(self.cv_datasets):
            self.run_fold(fold_idx=fold_idx, train_eval_pair=train_eval_pair)
