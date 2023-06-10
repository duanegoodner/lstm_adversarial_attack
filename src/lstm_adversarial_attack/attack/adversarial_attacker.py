import torch
import torch.nn as nn
import lstm_adversarial_attack.attack.attacker_helpers as ah
import lstm_adversarial_attack.attack.feature_perturber as fp


class AdversarialAttacker(nn.Module):
    def __init__(
        self,
        full_model: nn.Module,
        state_dict: dict,
        input_size: int,
        max_sequence_length: int,
        batch_size: int,
    ):
        super(AdversarialAttacker, self).__init__()

        self.feature_perturber = fp.FeaturePerturber(
            batch_size=batch_size,
            input_size=input_size,
            max_sequence_length=max_sequence_length
        )

        self.logit_no_dropout_model = ah.LogitNoDropoutModelBuilder(
            full_model=full_model,
            state_dict=state_dict
        ).build()

    @property
    def batch_size(self) -> int:
        return self.feature_perturber.batch_size

    @batch_size.setter
    def batch_size(self, new_batch_size: int):
        self.feature_perturber.batch_size = new_batch_size

    def forward(self, feature: torch.tensor) -> torch.tensor:
        perturbed_feature = self.feature_perturber(feature)
        logits = self.logit_no_dropout_model(perturbed_feature)
        return perturbed_feature, logits

