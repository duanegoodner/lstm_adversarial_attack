import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Callable

from adversarial_attacker import AdversarialAttacker
from attacker_helpers import TargetDatasetBuilder
from inferrer import StandardModelInferrer, StandardInferenceResults
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
                batch_size=self.attacker.feature_perturber.batch_size,
                collate_fn=self.collate_fn,
            ).build()
        else:
            return DataLoader(
                dataset=target_dataset,
                batch_size=self.attacker.feature_perturber.batch_size,
                collate_fn=self.collate_fn,
            )

    def attack_batch(
        self,
        indices: torch.tensor,
        orig_features: VariableLengthFeatures,
        orig_labels: torch.tensor,
        max_num_attempts: int,
    ):
        self.attacker.feature_perturber.reset_parameters()
        orig_features.features, orig_labels = orig_features.features.to(
            self.device
        ), orig_labels.to(self.device)
        attempt_counts = 0
        success_counts = torch.zeros(orig_labels.shape[0], dtype=torch.long)

        success_dataset_indices = torch.LongTensor()
        success_padded_perturbations = torch.FloatTensor()
        success_input_lengths = torch.LongTensor()

        # TODO fill in details to record successes
        for epoch in range(max_num_attempts):
            self.optimizer.zero_grad()
            perturbed_features, logits = self.attacker(orig_features)
            loss = (
                self.loss_fn(logits=logits, original_labels=orig_labels)
                + self.lambda_1 * self.attacker.feature_perturber.l1_loss()
            )
            in_batch_success_indices = (
                torch.argmax(input=logits, dim=1) != orig_labels
            )
            success_dataset_indices = torch.cat(
                (
                    success_dataset_indices,
                    torch.masked_select(
                        input=indices, mask=in_batch_success_indices.to("cpu")
                    ),
                ),
                dim=0,
            )

            success_padded_perturbations = torch.cat(
                (
                    success_padded_perturbations,
                    self.attacker.feature_perturber.perturbation[
                        torch.where(in_batch_success_indices)[0], :, :
                    ].to("cpu"),
                ),
                dim=0,
            )

            loss.backward()
            self.optimizer.step()

    def train_attacker(self):
        data_loader = self.build_data_loader()
        self.attacker.to(self.device)
        self.set_attacker_train_mode()
        for num_batches, (indices, orig_features, orig_labels) in enumerate(
            data_loader
        ):
            self.attack_batch(
                indices=indices,
                orig_features=orig_features,
                orig_labels=orig_labels,
                max_num_attempts=100,
            )
