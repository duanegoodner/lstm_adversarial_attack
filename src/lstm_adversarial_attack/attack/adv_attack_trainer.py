from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import lstm_adversarial_attack.attack.adversarial_attacker as aat
import lstm_adversarial_attack.attack.attack_data_structs as ads
import lstm_adversarial_attack.attack.attack_result_data_structs as ards
import lstm_adversarial_attack.attack.attacker_helpers as ath
import lstm_adversarial_attack.config_settings as lcs
import lstm_adversarial_attack.data_structures as ds
import lstm_adversarial_attack.dataset_with_index as dsi
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.weighted_dataloader_builder as wdb


class AdversarialAttackTrainer:
    """
    Finds adversarial example_data for a classification model
    """

    # TODO consolidate params that are part of hyperparams into single object
    #  (large number of constructor args is messy)
    def __init__(
        self,
        device: torch.device,
        model: nn.Module,
        attack_hyperparameters: ads.AttackHyperParameterSettings,
        state_dict: dict,
        # batch_size: int,
        epochs_per_batch: int,
        # kappa: float,
        # lambda_1: float,
        # optimizer_constructor: Callable,
        # optimizer_constructor_kwargs: dict,
        dataset: dsi.DatasetWithIndex | Subset,
        collate_fn: Callable,
        attack_misclassified_samples: bool,
        inference_batch_size: int = 128,
        use_weighted_data_loader: bool = False,
        checkpoint_interval: int = None,
        output_dir: Path = None,
    ):
        """

        :param device: device to run on
        :param model: original classification model to attack
        :param state_dict: state dict obtained from previous training of model
        :param epochs_per_batch: number of attack iterations to run per batch
        (https://arxiv.org/abs/1802.04822). Defines a margin by which alternate
        class logit value needs to exceed original class logit value in order
        to reduce loss function.
        to sparse perturbations)
        :param dataset: unfiltered dataset potential samples to use for attacks
        :param collate_fn: method used by dataloader to organize dataset
        elements into batches
        :param attack_misclassified_samples: whether to run attacks on
        samples that original model misclassifies
        :param inference_batch_size: batch size to use when running inference
        with full dataset and original model
        :param use_weighted_data_loader: during attack, do we use dataloader
        that oversamples from minority class?
        :param checkpoint_interval: number of attack batches per checkpoint
        :param output_dir: directory where checkpoint .pickles get saved
        """
        self.device = device
        self.model = model
        self.attack_hyperparameters = attack_hyperparameters
        self.state_dict = state_dict
        self.attacker = aat.AdversarialAttacker(
            full_model=model,
            state_dict=state_dict,
            input_size=19,
            max_sequence_length=lcs.MAX_OBSERVATION_HOURS,
            batch_size=2**self.attack_hyperparameters.log_batch_size,
        )
        # self.kappa = kappa
        # self.lambda_1 = lambda_1
        self.epochs_per_batch = epochs_per_batch
        self.loss_fn = ath.AdversarialLoss(
            kappa=self.attack_hyperparameters.kappa,
            lambda_1=self.attack_hyperparameters.lambda_1,
        )
        # self.optimizer_constructor = optimizer_constructor
        # self.optimizer_constructor_kwargs = optimizer_constructor_kwargs
        self.optimizer_constructor_kwargs = {
            "lr": self.attack_hyperparameters.learning_rate
        }
        self.optimizer = self.attack_hyperparameters.optimizer_constructor(
            params=self.attacker.parameters(), **self.optimizer_constructor_kwargs
        )
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.inference_batch_size = inference_batch_size
        self.attack_misclassified_samples = attack_misclassified_samples
        self.use_weighted_data_loader = use_weighted_data_loader
        self.latest_result = None
        self.checkpoint_interval = checkpoint_interval
        self.output_dir = output_dir
        if self.checkpoint_interval is not None:
            assert self.output_dir is not None

    @property
    def batch_size(self) -> int:
        """
        Convenience getter.
        :return: batch size of .attacker's FeaturePerturber
        """
        return self.attacker.feature_perturber.batch_size

    @property
    def max_seq_length(self) -> int:
        """
        Convenience getter.
        :return: Max sequence length of .attacker's feature perturber.
        Corresponds to num rows in perturbation tensor.
        """
        return self.attacker.feature_perturber.max_sequence_length

    @property
    def input_size(self) -> int:
        """
        Convenience getter.
        :return: Input size of .attacker's feature perturber. Corresponds to
        number of columns in perturbation tensor.
        """
        return self.attacker.feature_perturber.input_size

    @property
    def perturbation(self) -> torch.tensor:
        """
        Convenience getter.
        :return: .attacker's FeaturePerturber's perturbation tensor
        """
        return self.attacker.feature_perturber.perturbation

    def reset_perturbation(self):
        """
        Resets parameters perturbation tensor
        """
        self.attacker.feature_perturber.reset_parameters()

    def set_attacker_train_mode(self):
        """
        Sets attacker to train mode which sets module parameters requires
        grad to true. Turns off grad in the predictive model portion of the
        overall attacker + predictor model. Tried setting predictive model to
        eval() mode, but that prevented backprop of attacker params.
        """
        self.attacker.train()
        for param in self.attacker.logit_no_dropout_model.parameters():
            param.requires_grad = False

    def get_target_dataset(self) -> dsi.DatasetWithIndex | Subset:
        """
        Creates dataset of samples to be attacked. Typical params will
        result in dataset with all of the correctly predicted samples from
        orig_dataset.
        :return: either the full self.dataset or subset of it
        """
        target_dataset_builder = ath.TargetDatasetBuilder(
            device=self.device,
            model=self.attacker.logit_no_dropout_model,
            orig_dataset=self.dataset,
            collate_fn=self.collate_fn,
            batch_size=self.inference_batch_size,
        )
        return target_dataset_builder.build()

    def build_data_loader(self) -> DataLoader:
        """
        Creates a dataloader to be used during training.
        :return: a DataLoader (either "regular" or "weighted")
        """
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
    ) -> ards.EpochSuccesses:
        """
        Gets info for samples within batch that are successfully attacked
        during an attack epoch.
        :param epoch_num: epoch number within training session
        :param logits: Model outputs of each sample in batch. When
        training/using predictive model for normal use, these logits are the
        values fed into final activation function. But when running attack,
        we don't use the final activation.
        :param orig_labels: class predicted by original model (for each sample
        in batch)
        :param sample_losses: attack loss for each sample in batch
        :return:
        """
        if logits.dim() == 1:
            logits = logits[None, :]
        attack_is_successful = torch.argmax(input=logits, dim=1) != orig_labels
        successful_attack_indices = torch.where(attack_is_successful)[0]
        epoch_success_losses = sample_losses[successful_attack_indices]
        epoch_success_perturbations = self.perturbation[
            successful_attack_indices
        ]

        return ards.EpochSuccesses(
            epoch_num=epoch_num,
            batch_indices=successful_attack_indices,
            losses=epoch_success_losses.detach(),
            perturbations=epoch_success_perturbations.detach(),
        )

    def rebuild_attacker(self, batch_size: int):
        """
        Creates a new AdversarialAttacker. Used when current attacker batch
        size does not match size of current batch (e.g. final batch of dataset)
        :param batch_size:
        :type batch_size:
        """
        self.attacker = aat.AdversarialAttacker(
            full_model=self.model,
            state_dict=self.state_dict,
            input_size=19,
            max_sequence_length=lcs.MAX_OBSERVATION_HOURS,
            batch_size=batch_size,
        )
        self.optimizer = self.attack_hyperparameters.optimizer_constructor(
            params=self.attacker.parameters(),
            **self.optimizer_constructor_kwargs,
        )
        self.attacker.to(self.device)

    def apply_soft_bounded_threshold(
        self, orig_inputs: ds.VariableLengthFeatures
    ):
        """
        Applies thresholding perturbation for Iterative Soft Threshold
        Algorithm (ISTA) implementation of adversarial loss regularization.
        :param orig_inputs: batch of sample input features
        """
        perturbation_min = -1 * orig_inputs.features
        perturbation_max = (
            torch.ones_like(orig_inputs.features) - orig_inputs.features
        )

        # TODO Create setter methods so trainer does not need to
        #  directly access perturbations
        zero_mask = (
            torch.abs(self.attacker.feature_perturber.perturbation)
            <= self.attack_hyperparameters.lambda_1
        )
        self.attacker.feature_perturber.perturbation.data[zero_mask] = 0
        pos_mask = (
            self.attacker.feature_perturber.perturbation
            > self.attack_hyperparameters.lambda_1
        )
        self.attacker.feature_perturber.perturbation.data[
            pos_mask
        ] -= self.attack_hyperparameters.lambda_1
        neg_mask = (
            self.attacker.feature_perturber.perturbation
            < -1 * self.attack_hyperparameters.lambda_1
        )
        self.attacker.feature_perturber.perturbation.data[
            neg_mask
        ] += self.attack_hyperparameters.lambda_1
        clamped_perturbation = torch.clamp(
            input=self.attacker.feature_perturber.perturbation.data,
            min=perturbation_min,
            max=perturbation_max,
        )
        self.attacker.feature_perturber.perturbation.data.copy_(
            clamped_perturbation
        )

    # TODO Refactor. Method is too long.
    def attack_batch(
        self,
        indices: torch.tensor,
        orig_features: ds.VariableLengthFeatures,
        orig_labels: torch.tensor,
    ) -> ards.BatchResult:
        """
        Attacks batch of samples for self.num_epochs.
        :param indices: dataset indices of samples in batch
        :param orig_features: input features of samples in batch
        :param orig_labels: target model's class predictions
        :return: a BatchResult object containing first and best (lowest loss)
        EpochSuccess info for each sample. Samples with no adversarial example_data
        found are included in this but have orig initialized values that can
        be interpreted as no example found.
        """
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

        batch_result = ards.BatchResult(
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
        """
        Displays attack details to terminal
        """
        print(
            f"Running attacks with:\nbatch_size = {self.batch_size}\nkappa ="
            f" {self.attack_hyperparameters.kappa}\nlambda_1 ="
            f" {self.attack_hyperparameters.lambda_1}\noptimizer ="
            f" {self.attack_hyperparameters.optimizer_constructor}\noptimizer"
            " constructor kwargs ="
            f" {self.optimizer_constructor_kwargs}\nepochs per batch ="
            f" {self.epochs_per_batch}\nmax number of samples ="
            f" {len(self.dataset)}\n"
        )

    def save_checkpoint(
        self, trainer_result: ards.TrainerResult, num_batches: int
    ):
        """
        Saves TrainerResult, which contains info from all BatchResults, to
        .pickle file. (TrainerResult gets updated after at end of each
        batch loop in .train_attacker().)
        :param trainer_result:
        :param num_batches:
        :return:
        """
        checkpoint_path = rio.create_timestamped_filepath(
            parent_path=self.output_dir,
            file_extension="pickle",
            suffix=f"_{num_batches + 1}_batch_attack_result",
        )
        rio.ResourceExporter().export(
            resource=trainer_result, path=checkpoint_path
        )

    def train_attacker(self):
        """
        Trains AdversarialAttacker for all samples in batch. "Training" the
        attacker means searching for perturbations that minimize its loss
        function. Each sample that has adversarial example(s) found
        will have information for its first found example and its lowest loss
        (aka "best") example.
        :return: TrainerResult containing info on first and best adversarial
        example_data. See TrainerResult docstring for description of values stored
        for samples that have no example found.
        """
        data_loader = self.build_data_loader()
        self.attacker.to(self.device)
        self.set_attacker_train_mode()
        trainer_result = ards.TrainerResult(dataset=self.dataset)

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

            if self.checkpoint_interval is not None and (
                (num_batches + 1) % self.checkpoint_interval == 0
            ):
                self.save_checkpoint(
                    trainer_result=trainer_result, num_batches=num_batches
                )

        self.latest_result = trainer_result

        return trainer_result
