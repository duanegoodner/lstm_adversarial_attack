import sys
import torch
import torch.nn as nn
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from lstm_adversarial_attack.data_structures import VariableLengthFeatures


class FeaturePerturber(nn.Module):
    def __init__(
        self,
        initial_batch_size: int,
        input_size: int,
        max_sequence_length: int,
    ):
        super(FeaturePerturber, self).__init__()
        self._batch_size = initial_batch_size
        self.input_size = input_size
        self.max_sequence_length = max_sequence_length
        self.perturbation = nn.Parameter(
            torch.zeros((initial_batch_size, max_sequence_length, input_size))
        )

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_batch_size: int):
        orig_device = self.perturbation.device
        self._batch_size = new_batch_size
        self.perturbation = nn.Parameter(
            torch.zeros(
                (new_batch_size, self.max_sequence_length, self.input_size)
            )
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
