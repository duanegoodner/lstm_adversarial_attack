import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import Subset
from typing import Callable
import lstm_adversarial_attack.attack.inferrer as infr
import lstm_adversarial_attack.x19_mort_general_dataset as xmd


@dataclass
class TargetModel:
    full_model: nn.Module
    full_model_state_dict: dict
    logit_out_no_dropout: nn.Module


# TODO Is it necessary to include state_dict? Or can we guarantee it gets
#  loaded into model and just use model?
class TargetModelBuilder:
    def __init__(self, full_model: nn.Module, state_dict: dict):
        self.full_model = full_model
        self.state_dict = state_dict

    def build_logit_out_no_dropout(self) -> nn.Module:
        new_module_list = [
            val if type(val) != nn.Dropout else nn.Dropout(0)
            for key, val in list(self.full_model._modules.items())[:-1]
        ]
        logit_no_dropout_model = nn.Sequential(*new_module_list)
        logit_no_dropout_model.load_state_dict(self.state_dict, strict=False)
        return logit_no_dropout_model

    def build(self) -> TargetModel:
        return TargetModel(
            full_model=self.full_model,
            full_model_state_dict=self.state_dict,
            logit_out_no_dropout=self.build_logit_out_no_dropout(),
        )


class LogitNoDropoutModelBuilder:
    def __init__(self, full_model: nn.Module, state_dict: dict):
        self.full_model = full_model
        self.state_dict = state_dict

    def build(self) -> nn.Module:
        new_module_list = [
            val if type(val) != nn.Dropout else nn.Dropout(0)
            for key, val in list(self.full_model._modules.items())[:-1]
        ]
        logit_no_dropout_model = nn.Sequential(*new_module_list)
        logit_no_dropout_model.load_state_dict(self.state_dict, strict=False)
        return logit_no_dropout_model


class TargetDatasetBuilder:
    def __init__(
        self,
        device: torch.device,
        model: nn.Module,
        orig_dataset: xmd.DatasetWithIndex | xmd.X19MGeneralDatasetWithIndex,
        collate_fn: Callable,
        batch_size: int = 128,
        include_misclassified_examples: bool = False,
    ):
        self.device = device
        self.model = model
        self.orig_dataset = orig_dataset
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.include_misclassified_examples = include_misclassified_examples

    def get_orig_predictions(self) -> infr.StandardInferenceResults:
        inferrer = infr.StandardModelInferrer(
            device=self.device,
            model=self.model,
            dataset=self.orig_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
        )
        return inferrer.infer()

    def build(self) -> xmd.DatasetWithIndex | Subset:
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
    def __init__(self, kappa: float, lambda_1: float):
        super(AdversarialLoss, self).__init__()
        self.kappa = kappa
        self.lambda_1 = lambda_1

    def forward(
        self,
        logits: torch.tensor,
        perturbations: torch.tensor,
        original_labels: torch.tensor,
    ) -> torch.tensor:

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
        l1_losses = self.lambda_1 * torch.linalg.matrix_norm(
            perturbations, ord=1
        )
        sample_net_losses = torch.squeeze(logit_losses) + l1_losses
        mean_loss = torch.mean(sample_net_losses)
        return mean_loss, sample_net_losses

