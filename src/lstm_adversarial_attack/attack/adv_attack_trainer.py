from datetime import datetime
import numpy as np
import time
import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import DataLoader

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
        optimizer: torch.optim.Optimizer,
        data_loader: DataLoader,
        output_dir: Path,
    ):
        self.device = device
        self.attacker = attacker
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.output_dir = output_dir

    def set_attacker_train_mode(self):
        self.attacker.train()
        for param in self.attacker.logitout_model.parameters():
            param.requires_grad = False

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
            loss = self.loss_fn(logits=logits, original_labels=orig_labels)
            regularized_loss = loss + torch.linalg.matrix_norm(
                self.attacker.feature_perturber.perturbation
            )
            regularized_loss.backward()
            self.optimizer.step()

    def train_attacker(self):
        self.attacker.to(self.device)
        self.set_attacker_train_mode()
        for num_batches, (indices, orig_features, orig_labels) in enumerate(
            self.data_loader
        ):
            self.attack_batch(
                indices=indices,
                orig_features=orig_features,
                orig_labels=orig_labels,
                max_num_attempts=5,
            )
