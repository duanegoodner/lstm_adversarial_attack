from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import Subset

import lstm_adversarial_attack.attack.inferrer as infr
import lstm_adversarial_attack.x19_mort_general_dataset as xmd


@dataclass
class TargetModel:
    """
    Container for model to be attacked.
    :param full_model: Binary classification model with activation of its
    final output layer. May contain dropout layers.
    :param full_model_state_dict: state dict of parameters obtained from
    training full_model
    :param :logit_out_no_dropout: model created by starting with full_model,
    then removing its final activation function and setting dropout prob of
    any/all dropout layers to 0.
    """
    full_model: nn.Module
    full_model_state_dict: dict
    logit_out_no_dropout: nn.Module


class LogitNoDropoutModelBuilder:
    """
    Creates a model of the form needed for TargetModel.logit_out_no_dropout
    """
    def __init__(self, full_model: nn.Module, state_dict: dict):
        """
        :param full_model: Classification model with a final activation layer
        and possibly with dropout layer(s)
        :param state_dict: state dict obtained from previous training of
        full_model
        """
        self.full_model = full_model
        self.state_dict = state_dict

    def build(self) -> nn.Module:
        """
        Builds a model (based on self.full_model) with no final activation
        layer and any/all dropout probs set to zero
        :return: Classification model with logit as final output, no dropout,
        and state dict loaded.
        """
        new_module_list = [
            val if type(val) != nn.Dropout else nn.Dropout(0)
            for key, val in list(self.full_model._modules.items())[:-1]
        ]
        logit_no_dropout_model = nn.Sequential(*new_module_list)
        logit_no_dropout_model.load_state_dict(self.state_dict, strict=False)
        return logit_no_dropout_model


class TargetDatasetBuilder:
    """
    Creates dataset to be used during attack of a model. Default behavior is to
    generate a Subset of full dataset consisting of all samples correctly
    classified model to be attacked.
    """
    def __init__(
        self,
        device: torch.device,
        model: nn.Module,
        orig_dataset: xmd.DatasetWithIndex | xmd.X19MGeneralDatasetWithIndex,
        collate_fn: Callable,
        batch_size: int = 128,
        include_misclassified_examples: bool = False,
    ):
        """
        :param device: device to run on
        :param model: classification model to be attacked
        :param orig_dataset: dataset of possible samples to use during attack
        :param collate_fn: organizes dataset elements into batches (when
        running dataset through model for inference to determine correctly
        classified samples)
        :param batch_size: number of samples per batch during inference
        :param include_misclassified_examples: whether or not to include
        misclassified samples in final/filtered dataset.
        """
        self.device = device
        self.model = model
        self.orig_dataset = orig_dataset
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.include_misclassified_examples = include_misclassified_examples

    def get_orig_predictions(self) -> infr.StandardInferenceResults:
        """
        Uses .model to classify all samples in original dataset
        :return: dataclass object containing inference results
        """
        inferrer = infr.StandardModelInferrer(
            device=self.device,
            model=self.model,
            dataset=self.orig_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
        )
        return inferrer.infer()

    def build(self) -> xmd.DatasetWithIndex | Subset:
        """
        Generates final dataset using inference results from .get_predictions()
        :return: a dataset (to be used for attack)
        """
        orig_predictions = self.get_orig_predictions()
        full_target_dataset = xmd.X19MGeneralDatasetWithIndex(
            measurements=self.orig_dataset.measurements,
            in_hosp_mort=orig_predictions.y_pred,
        )
        if not self.include_misclassified_examples:
            return Subset(
                dataset=full_target_dataset,
                indices=orig_predictions.correct_prediction_indices,
            )
        else:
            return full_target_dataset


class AdversarialLoss(nn.Module):
    """
    Module with loss function used during adversarial attack
    """
    def __init__(self, kappa: float, lambda_1: float):
        """
        :param kappa: Parameter from Equation 1 in Sun et al
        (https://arxiv.org/abs/1802.04822). Defines a margin by which alternate
        class logit value needs to exceed original class logit value in order
        to reduce loss function.
        :param lambda_1: L1 regularization constant (to encourage finding
        adversarial example_data caused by sparse perturbations).
        """
        super(AdversarialLoss, self).__init__()
        self.kappa = kappa
        self.lambda_1 = lambda_1

    def forward(
        self,
        logits: torch.tensor,
        perturbations: torch.tensor,
        original_labels: torch.tensor,
    ) -> torch.tensor:
        """

        :param logits: output of the classifier (with activation removed)
        :param perturbations: values added to original input features
        :param original_labels: predicted class when original sample features
        are run through model under attack
        :return: tensor of loss values (one value per sample in batch)
        """

        # rest of function assumes 2-D logit tensor
        if logits.dim() == 1:
            logits = logits[None, :]

        orig_predicted_logit = logits[
            torch.arange(logits.size(0)), original_labels
        ]
        orig_not_predicted_logit = logits[
            torch.arange(logits.size(0)), (~original_labels.bool()).int()
        ]

        logit_deltas = orig_predicted_logit - orig_not_predicted_logit
        logit_losses = torch.max(logit_deltas, -1 * torch.tensor(self.kappa))

        # We attempt to solve regularized loss function by gradient descent,
        # but will likely apply ISTA on top of this (see
        # AdversarialAttackTrainer.apply_soft_bounded_threshold method). If we
        # use gradient descent without ISTA, perturbations become very small,
        # but don't go to zero. This consistent with numerous reports that
        # ISTA typically converges faster than gradient descent.
        l1_losses = self.lambda_1 * torch.linalg.matrix_norm(
            perturbations, ord=1
        )
        sample_net_losses = torch.squeeze(logit_losses) + l1_losses
        mean_loss = torch.mean(sample_net_losses)
        return mean_loss, sample_net_losses

