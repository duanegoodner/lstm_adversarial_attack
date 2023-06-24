import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import Callable

import lstm_adversarial_attack.attack.adversarial_attacker as aat
import lstm_adversarial_attack.attack.attacker_helpers as ath
import lstm_adversarial_attack.attack.attack_result_data_structs as ads
import lstm_adversarial_attack.config_settings as lcs
import lstm_adversarial_attack.dataset_with_index as dsi
import lstm_adversarial_attack.data_structures as ds
import lstm_adversarial_attack.weighted_dataloader_builder as wdb


class AdversarialAttackTrainer:
    def __init__(
        self,
        device: torch.device,
        model: nn.Module,
        state_dict: dict,
        batch_size: int,
        epochs_per_batch: int,
        kappa: float,
        lambda_1: float,
        optimizer_constructor: Callable,
        optimizer_constructor_kwargs: dict,
        dataset: dsi.DatasetWithIndex | Subset,
        collate_fn: Callable,
        attack_misclassified_samples: bool,
        inference_batch_size: int = 128,
        use_weighted_data_loader: bool = False,
    ):
        self.device = device
        self.model = model
        self.state_dict = state_dict
        self.attacker = aat.AdversarialAttacker(
            full_model=model,
            state_dict=state_dict,
            input_size=19,
            max_sequence_length=lcs.MAX_OBSERVATION_HOURS,
            batch_size=batch_size,
        )
        self.kappa = kappa
        self.lambda_1 = lambda_1
        self.epochs_per_batch = epochs_per_batch
        self.loss_fn = ath.AdversarialLoss(kappa=kappa, lambda_1=lambda_1)
        self.optimizer_constructor = optimizer_constructor
        self.optimizer_constructor_kwargs = optimizer_constructor_kwargs
        self.optimizer = optimizer_constructor(
            params=self.attacker.parameters(), **optimizer_constructor_kwargs
        )
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.inference_batch_size = inference_batch_size
        self.attack_misclassified_samples = attack_misclassified_samples
        self.use_weighted_data_loader = use_weighted_data_loader
        self.latest_result = None

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

    def get_target_dataset(self) -> dsi.DatasetWithIndex | Subset:
        target_dataset_builder = ath.TargetDatasetBuilder(
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
            return wdb.WeightedDataLoaderBuilder(
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
    ) -> ads.EpochSuccesses:
        if logits.dim() == 1:
            logits = logits[None, :]
        attack_is_successful = torch.argmax(input=logits, dim=1) != orig_labels
        successful_attack_indices = torch.where(attack_is_successful)[0]
        epoch_success_losses = sample_losses[successful_attack_indices]
        epoch_success_perturbations = self.perturbation[
            successful_attack_indices
        ]

        return ads.EpochSuccesses(
            epoch_num=epoch_num,
            batch_indices=successful_attack_indices,
            losses=epoch_success_losses.detach(),
            perturbations=epoch_success_perturbations.detach(),
        )

    def rebuild_attacker(self, batch_size: int):
        self.attacker = aat.AdversarialAttacker(
            full_model=self.model,
            state_dict=self.state_dict,
            input_size=19,
            max_sequence_length=lcs.MAX_OBSERVATION_HOURS,
            batch_size=batch_size,
        )
        self.optimizer = self.optimizer_constructor(
            params=self.attacker.parameters(),
            **self.optimizer_constructor_kwargs,
        )
        self.attacker.to(self.device)

    def apply_soft_bounded_threshold(
        self, orig_inputs: ds.VariableLengthFeatures
    ):
        perturbation_min = -1 * orig_inputs.features
        perturbation_max = (
            torch.ones_like(orig_inputs.features) - orig_inputs.features
        )

        # TODO Create setter methods so trainer does not need to
        #  directly access perturbations
        zero_mask = torch.abs(
            self.attacker.feature_perturber.perturbation <= self.lambda_1
        )
        self.attacker.feature_perturber.perturbation.data[zero_mask] = 0
        pos_mask = self.attacker.feature_perturber.perturbation > self.lambda_1
        self.attacker.feature_perturber.perturbation.data[
            pos_mask
        ] -= self.lambda_1
        neg_mask = (
            self.attacker.feature_perturber.perturbation < -1 * self.lambda_1
        )
        self.attacker.feature_perturber.perturbation.data[
            neg_mask
        ] += self.lambda_1
        clamped_perturbation = torch.clamp(
            input=self.attacker.feature_perturber.perturbation.data,
            min=perturbation_min,
            max=perturbation_max,
        )
        self.attacker.feature_perturber.perturbation.data.copy_(
            clamped_perturbation
        )

    def attack_batch(
        self,
        indices: torch.tensor,
        orig_features: ds.VariableLengthFeatures,
        orig_labels: torch.tensor,
    ):
        # if batch size less than size of perturbations dim 1, re-build
        # FeaturePerturber to match current batch size
        if indices.shape[0] != self.attacker.batch_size:
            self.rebuild_attacker(batch_size=indices.shape[0])

        # if max sequence length in batch less than perturber's max sequence
        # length, pad entire batch to match perturber
        if (
            orig_features.features.shape[1]
            < self.attacker.feature_perturber.perturbation.shape[1]
        ):
            num_rows_to_add = (
                self.attacker.feature_perturber.perturbation.shape[1]
                - orig_features.features.shape[1]
            )
            orig_features.features = torch.cat(
                (
                    orig_features.features,
                    torch.zeros(
                        (
                            orig_features.features.shape[0],
                            num_rows_to_add,
                            orig_features.features.shape[2],
                        ),
                        dtype=torch.float32,
                    ),
                ),
                dim=1,
            )

        self.reset_perturbation()
        orig_features.features, orig_labels = orig_features.features.to(
            self.device
        ), orig_labels.to(self.device)

        batch_result = ads.BatchResult(
            dataset_indices=indices,
            input_seq_lengths=orig_features.lengths,
            max_seq_length=self.max_seq_length,
            input_size=self.input_size,
        )

        for epoch in range(self.epochs_per_batch):
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
                sample_losses=sample_losses,
            )
            batch_result.update(epoch_successes=epoch_successes)

            mean_loss.backward()
            self.optimizer.step()
            self.apply_soft_bounded_threshold(orig_inputs=orig_features)

        return batch_result

    def display_attack_info(self):
        print(
            f"Running attacks with:\nbatch_size = {self.batch_size}\nkappa ="
            f" {self.kappa}\nlambda_1 = {self.lambda_1}\noptimizer ="
            f" {self.optimizer_constructor}\noptimizer constructor kwargs ="
            f" {self.optimizer_constructor_kwargs}\nepochs per batch ="
            f" {self.epochs_per_batch}\nmax number of samples ="
            f" {len(self.dataset)}\n"
        )

    def train_attacker(self):
        data_loader = self.build_data_loader()
        self.attacker.to(self.device)
        self.set_attacker_train_mode()
        trainer_result = ads.TrainerResult(dataset=self.dataset)

        self.display_attack_info()

        for num_batches, (indices, orig_features, orig_labels) in enumerate(
            data_loader
        ):
            print(f"Running batch {num_batches}")
            batch_result = self.attack_batch(
                indices=indices,
                orig_features=orig_features,
                orig_labels=orig_labels,
            )

            trainer_result.update(batch_result=batch_result)

        self.latest_result = trainer_result

        return trainer_result
