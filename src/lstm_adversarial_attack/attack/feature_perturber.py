import sys
import torch
import torch.nn as nn
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from lstm_adversarial_attack.data_structures import VariableLengthFeatures


class FeaturePerturber(nn.Module):
    def __init__(
        self,
        batch_size: int = 2,
        input_size: int = 3,
        max_sequence_length: int = 4,
    ):
        super(FeaturePerturber, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.max_sequence_length = max_sequence_length
        self.perturbation = nn.Parameter(
            torch.zeros((max_sequence_length, input_size))
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
