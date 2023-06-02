from datetime import datetime
import numpy as np
import time
import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from typing import Callable

from adversarial_attacker import AdversarialAttacker
from attacker_helpers import AdversarialLoss
from lstm_adversarial_attack.dataset_with_index import DatasetWithIndex
from lstm_adversarial_attack.data_structures import VariableLengthFeatures
from lstm_sun_2018_logit_out import LSTMSun2018Logit
import lstm_adversarial_attack.resource_io as rio
from feature_perturber import FeaturePerturber
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
        dataset: Dataset,
        output_dir: Path,
        collate_fn: Callable,
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
        self.use_weighted_data_loader = use_weighted_data_loader

    def set_attacker_train_mode(self):
        self.attacker.train()
        for param in self.attacker.logitout_model.parameters():
            param.requires_grad = False

    def build_data_loader(self) -> DataLoader:
        if self.use_weighted_data_loader:
            return WeightedDataLoaderBuilder(
                dataset=self.dataset,
                batch_size=self.attacker.feature_perturber.batch_size,
                collate_fn=self.collate_fn,
            ).build()
        else:
            return DataLoader(
                dataset=self.dataset,
                batch_size=self.attacker.feature_perturber.batch_size,
                collate_fn=self.collate_fn,
            )

    def get_successful_examples(
        self,
        dataset_indices: torch.tensor,
        orig_features: torch.tensor,
        logits: torch.tensor,
        orig_labels: torch.tensor,
    ):
        in_batch_success_indices = (
            torch.argmax(input=logits, dim=1) != orig_labels
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

        # TODO fill in details to record successes
        for epoch in range(max_num_attempts):
            self.optimizer.zero_grad()
            perturbed_features, logits = self.attacker(orig_features)
            loss = (
                self.loss_fn(logits=logits, original_labels=orig_labels)
                + self.lambda_1 * self.attacker.feature_perturber.l1_loss()
            )
            # regularized_loss = loss + torch.linalg.matrix_norm(
            #     self.attacker.feature_perturber.perturbation
            # )
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
