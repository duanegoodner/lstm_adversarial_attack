import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Callable

from adversarial_attacker import AdversarialAttacker
from attack_result_data_structs import (
    EpochSuccesses,
    BatchResult,
    TrainerResult
)
from attacker_helpers import TargetDatasetBuilder
from lstm_adversarial_attack.dataset_with_index import DatasetWithIndex
from lstm_adversarial_attack.data_structures import VariableLengthFeatures
from lstm_adversarial_attack.weighted_dataloader_builder import (
    WeightedDataLoaderBuilder,
)


class AdversarialAttackTrainer:
    def __init__(
        self,
        device: torch.device,
        attacker: AdversarialAttacker,
        loss_fn: nn.Module,
        lambda_1: float,
        optimizer: torch.optim.Optimizer,
        dataset: DatasetWithIndex,
        output_dir: Path,
        collate_fn: Callable,
        inference_batch_size: int,
        attack_misclassified_samples: bool = False,
        use_weighted_data_loader: bool = False,
    ):
        self.device = device
        self.attacker = attacker
        self.loss_fn = loss_fn
        self.lambda_1 = lambda_1
        self.optimizer = optimizer
        self.dataset = dataset
        self.output_dir = output_dir
        self.collate_fn = collate_fn
        self.inference_batch_size = inference_batch_size
        self.attack_misclassified_samples = attack_misclassified_samples
        self.use_weighted_data_loader = use_weighted_data_loader

    @property
    def batch_size(self) -> int:
        return self.attacker.feature_perturber.batch_size

    @property
    def max_seq_length(self) -> int:
        return self.attacker.feature_perturber.max_sequence_length

    @property
    def input_size(self) -> int:
        return self.attacker.feature_perturber.input_size

    @property
    def perturbation(self) -> torch.tensor:
        return self.attacker.feature_perturber.perturbation

    def reset_perturbation(self):
        self.attacker.feature_perturber.reset_parameters()

    def set_attacker_train_mode(self):
        self.attacker.train()
        for param in self.attacker.logit_no_dropout_model.parameters():
            param.requires_grad = False

    def get_target_dataset(self) -> DatasetWithIndex | Subset:
        target_dataset_builder = TargetDatasetBuilder(
            device=self.device,
            model=self.attacker.logit_no_dropout_model,
            orig_dataset=self.dataset,
            collate_fn=self.collate_fn,
            batch_size=self.inference_batch_size,
        )
        return target_dataset_builder.build()

    def build_data_loader(self) -> DataLoader:
        target_dataset = self.get_target_dataset()
        if self.use_weighted_data_loader:
            return WeightedDataLoaderBuilder(
                dataset=target_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
            ).build()
        else:
            return DataLoader(
                dataset=target_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
            )

    def get_epoch_successes(
        self,
        epoch_num: int,
        logits: torch.tensor,
        orig_labels: torch.tensor,
        sample_losses: torch.tensor,
    ) -> EpochSuccesses:
        attack_is_successful = torch.argmax(input=logits, dim=1) != orig_labels
        successful_attack_indices = torch.where(attack_is_successful)[0]
        epoch_success_losses = sample_losses[successful_attack_indices]
        epoch_success_perturbations = self.perturbation[
            successful_attack_indices
        ]

        return EpochSuccesses(
            epoch_num=epoch_num,
            batch_indices=successful_attack_indices,
            losses=epoch_success_losses,
            perturbations=epoch_success_perturbations
        )

    def attack_batch(
        self,
        indices: torch.tensor,
        orig_features: VariableLengthFeatures,
        orig_labels: torch.tensor,
        max_num_attempts: int,
    ):
        self.reset_perturbation()
        orig_features.features, orig_labels = orig_features.features.to(
            self.device
        ), orig_labels.to(self.device)

        batch_result = BatchResult(
            initial_device=self.device,
            dataset_indices=indices,
            max_seq_length=self.max_seq_length,
            input_size=self.input_size,
        )

        for epoch in range(max_num_attempts):
            self.optimizer.zero_grad()
            perturbed_features, logits = self.attacker(orig_features)
            mean_loss, sample_losses = self.loss_fn(
                logits=logits,
                perturbations=self.perturbation,
                original_labels=orig_labels,
            )

            # Run this BEFORE optimizer.step() which changes perturbations
            epoch_successes = self.get_epoch_successes(
                epoch_num=epoch,
                logits=logits,
                orig_labels=orig_labels,
                sample_losses=sample_losses
            )
            batch_result.update(epoch_successes=epoch_successes)

            mean_loss.backward()
            self.optimizer.step()

        return batch_result


    def train_attacker(self):
        data_loader = self.build_data_loader()
        self.attacker.to(self.device)
        self.set_attacker_train_mode()
        trainer_result = TrainerResult()

        for num_batches, (indices, orig_features, orig_labels) in enumerate(
            data_loader
        ):
            print(f"Running batch {num_batches}")
            batch_result = self.attack_batch(
                indices=indices,
                orig_features=orig_features,
                orig_labels=orig_labels,
                max_num_attempts=100,
            )

            trainer_result.update(batch_result=batch_result.to("cpu"))

        return trainer_result


