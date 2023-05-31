import numpy as np
import torch.nn as nn
import torch.optim
from pathlib import Path
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader, Dataset
import standard_model_trainer_old as smt
import standard_trainable_classifier as stc
from weighted_dataloader_builder import (
    DataLoaderBuilder,
    WeightedDataLoaderBuilder,
)


class CrossValidator:
    def __init__(
        self,
        model: stc.StandardTrainableClassifier,
        dataset: Dataset,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        # trainer: smt.StandardModelTrainer,
        num_folds: int,
        batch_size: int,
        epochs_per_fold: int,
        max_global_epochs: int,
        save_checkpoints: bool,
        checkpoints_dir: Path = Path.cwd(),
        dataloader_builder: DataLoaderBuilder = WeightedDataLoaderBuilder(),
    ):
        self.dataset = dataset
        # use for evals at checkpoint after each global epoch
        self.full_dataloader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False
        )
        self.num_folds = num_folds
        self.fold_generator = KFold(n_splits=num_folds, shuffle=True)
        self.batch_size = batch_size
        self.epochs_per_fold = epochs_per_fold
        self.max_global_epochs = max_global_epochs
        self.dataloader_builder = dataloader_builder
        self.save_checkpoints = save_checkpoints
        self.trainer = smt.StandardModelTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            save_checkpoints=False,  # let cross validator decide when to save
            checkpoint_dir=checkpoints_dir,
        )
        self.loss_log = []

    @property
    def dataset_size(self) -> int:
        return len(self.dataset)

    def train_fold(self, train_indices: np.ndarray):
        train_split = Subset(dataset=self.dataset, indices=train_indices)
        train_dataloader = self.dataloader_builder.build(
            dataset=train_split, batch_size=self.batch_size
        )
        self.trainer.train_model(
            train_dataloader=train_dataloader,
            num_epochs=self.epochs_per_fold,
            loss_log=self.loss_log,
        )

    def evaluate_fold(self, validation_indices: np.ndarray):
        validation_split = Subset(
            dataset=self.dataset, indices=validation_indices
        )
        validation_dataloader = DataLoader(
            dataset=validation_split, batch_size=self.batch_size, shuffle=True
        )
        self.trainer.evaluate_model(test_dataloader=validation_dataloader)

    def run_global_epoch(self):
        for fold_idx, (train_indices, validation_indices) in enumerate(
            self.fold_generator.split(range(self.dataset_size))
        ):
            self.train_fold(train_indices=train_indices)
            self.evaluate_fold(validation_indices=validation_indices)

    def run(self):
        for global_epoch in range(self.max_global_epochs):
            self.run_global_epoch()
            if self.save_checkpoints:
                metrics = self.trainer.evaluate_model(
                    test_dataloader=self.full_dataloader
                )
                self.trainer.save_checkpoint(
                    epoch_num=global_epoch + 1,
                    loss=self.loss_log[-1],
                    metrics=metrics
                )
        # be sure to checkpoint after all done even if not done after each
        # global epoch
        metrics = self.trainer.evaluate_model(
            test_dataloader=self.full_dataloader
        )
        final_checkpoint = self.trainer.save_checkpoint(
            epoch_num=self.max_global_epochs,
            loss=self.loss_log[-1],
            metrics=metrics
        )
        print(f"Model parameters saved in file: {final_checkpoint.name}")

