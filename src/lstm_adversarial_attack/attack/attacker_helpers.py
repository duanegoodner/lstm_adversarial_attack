import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Callable
from adversarial_attacker import AdversarialAttacker
from feature_perturber import FeaturePerturber

from lstm_adversarial_attack.weighted_dataloader_builder import (
    WeightedDataLoaderBuilder,
)


class AdversarialAttackerBuilder:
    def __init__(
        self,
        full_model: nn.Module,
        state_dict: dict,
        dataset: Dataset,
        batch_size: int,
        input_size: int,
        max_sequence_length: int,
        use_weighted_data_loader: bool = False,
    ):
        self.full_model = full_model
        self.state_dict = state_dict
        self.dataset = dataset
        self.batch_size = batch_size
        self.input_size = input_size
        self.max_sequence_length = max_sequence_length
        self.use_weighted_data_loader = use_weighted_data_loader

    # TODO try to use public full_model.modules() instead of ._modules (but
    #  need to figure out details of modules() return value)
    def create_logit_no_dropout_model(self) -> nn.Sequential:
        new_module_list = [
            val if type(val) != nn.Dropout else nn.Dropout(0)
            for key, val in list(self.full_model._modules.items())[:-1]
        ]
        logit_no_dropout_model = nn.Sequential(*new_module_list)
        logit_no_dropout_model.load_state_dict(self.state_dict, strict=False)
        return logit_no_dropout_model

    def build(self) -> AdversarialAttacker:
        logit_no_dropout_model = self.create_logit_no_dropout_model()
        feature_perturber = FeaturePerturber(
            batch_size=self.batch_size,
            input_size=self.input_size,
            max_sequence_length=self.max_sequence_length,
        )
        attacker = AdversarialAttacker(
            feature_perturber=feature_perturber,
            logitout_model=logit_no_dropout_model,
        )

        return attacker


class AdversarialLoss(nn.Module):
    def __init__(self, kappa: float):
        super(AdversarialLoss, self).__init__()
        self.kappa = kappa

    def forward(
        self, logits: torch.tensor, original_labels: torch.tensor
    ) -> torch.tensor:
        logit_deltas = logits.gather(
            1, original_labels.view(-1, 1)
        ) - logits.gather(1, (~original_labels.view(-1, 1).bool()).long())
        max_elements = torch.max(logit_deltas, -1 * torch.tensor(self.kappa))
        return torch.mean(max_elements)


# loss_fn = AdversarialLoss(kappa=0.0)
# my_logits = torch.tensor([[0.5, 0.4], [0.3, 0.5]], dtype=torch.float32)
# my_original_labels = torch.tensor([0, 1], dtype=torch.long)
#
# result = loss_fn(my_logits, my_original_labels)
