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
        model: nn.Module,
        batch_size: int,
        optimizer_name: str,
        learning_rate: float,
        num_folds: int,
        epochs_per_fold: int,
        fold_class: Callable = StratifiedKFold,
        kfold_random_seed: int = lcs.TUNER_KFOLD_RANDOM_SEED,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        cv_mean_metrics_of_interest: tuple[
            str
        ] = lcs.TUNER_CV_MEAN_METRICS_OF_INTEREST,
        output_dir: Path = None,
        save_fold_info: bool = True,
    ):
        self.device = device
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.model = model
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.num_folds = num_folds
        # self.num_cv_epochs = num_cv_epochs
        self.epochs_per_fold = epochs_per_fold
        self.fold_class = fold_class
        self.kfold_random_seed = kfold_random_seed
        self.cv_datasets = self.create_datasets()
        self.loss_fn = loss_fn
        self.cv_mean_metrics_of_interest = cv_mean_metrics_of_interest
        self.output_dir = output_dir
        self.tensorboard_output_dir = self.output_dir / "tensorboard"
        self.summary_writer = SummaryWriter(str(self.tensorboard_output_dir))
        self.trainer_checkpoint_dir = self.output_dir / "checkpoints_trainer"
        self.exporter = rio.ResourceExporter()
        self.save_fold_info = save_fold_info

    @classmethod
    def from_optuna_completed_trial_obj(
        cls,

    ):

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

    def initialize_model(self):
        for name, param in self.model.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def create_trainer(
        self, fold_idx: int, train_eval_pair: tuh.TrainEvalDatasetPair
    ) -> smt.StandardModelTrainer:
        self.initialize_model()

        train_loader = wdb.WeightedDataLoaderBuilder(
            dataset=train_eval_pair.train,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        ).build()

        validation_loader = DataLoader(
            dataset=train_eval_pair.validation,
            # TODO make validation batch size a config_settings value
            batch_size=128,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

        trainer = smt.StandardModelTrainer(
            train_device=self.device,
            eval_device=self.device,
            model=self.model,
            loss_fn=self.loss_fn,
            optimizer=getattr(torch.optim, self.optimizer_name)(
                self.model.parameters(), lr=self.learning_rate
            ),
            train_loader=train_loader,
            test_loader=validation_loader,
            summary_writer=self.summary_writer,
            summary_writer_group="cross_validation",
            summary_writer_subgroup=f"fold_{fold_idx}",
            checkpoint_dir=self.trainer_checkpoint_dir,
        )

        return trainer

    def run_fold(
        self, fold_idx: int, train_eval_pair: tuh.TrainEvalDatasetPair
    ):
        self.initialize_model()
        trainer = self.create_trainer(
            fold_idx=fold_idx, train_eval_pair=train_eval_pair
        )
        trainer.run_train_eval_cycles(
            num_cycles=1,
            epochs_per_cycle=self.epochs_per_fold,
            save_checkpoints=True,
        )

    def run_all_folds(self):
        for fold_idx, train_eval_pair in enumerate(self.cv_datasets):
            self.run_fold(fold_idx=fold_idx, train_eval_pair=train_eval_pair)
