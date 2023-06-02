import torch
import torch.nn as nn
from feature_perturber import FeaturePerturber


class  AdversarialAttacker(nn.Module):
    def __init__(
        self,
        feature_perturber: FeaturePerturber,
        logitout_model: nn.Module
    ):
        super(AdversarialAttacker, self).__init__()
        self.feature_perturber = feature_perturber
        self.logitout_model = logitout_model
        # self.to(self._device)

    def forward(self, feature: torch.tensor) -> torch.tensor:
        perturbed_feature = self.feature_perturber(feature)
        logits = self.logitout_model(perturbed_feature)
        return perturbed_feature, logits