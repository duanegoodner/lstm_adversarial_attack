import sys
import torch
import torch.nn as nn
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from lstm_adversarial_attack.data_structures import VariableLengthFeatures


class FeaturePerturber(nn.Module):
    def __init__(
        self,
        batch_size: int,
        input_size: int,
        max_sequence_length: int,
    ):
        super(FeaturePerturber, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.max_sequence_length = max_sequence_length
        self.perturbation = nn.Parameter(
            torch.zeros((batch_size, max_sequence_length, input_size))
        )

    def reset_parameters(self):
        if self.perturbation.grad is not None:
            self.perturbation.grad.zero_()
        nn.init.zeros_(self.perturbation)

    def forward(
        self, inputs: VariableLengthFeatures
    ) -> VariableLengthFeatures:

        return VariableLengthFeatures(
            features=inputs.features + self.perturbation,
            lengths=inputs.lengths,
        )

    # def l1_loss(self) -> torch.tensor:
    #     return torch.mean(torch.linalg.matrix_norm(self.perturbation))
