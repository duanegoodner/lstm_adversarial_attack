import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parent.parent.parent))
# import lstm_adversarial_attack.data_structures as ds
# import lstm_adversarial_attack.model.model_data_structs as mds
import lstm_adversarial_attack.model.lstm_model_stc as lms


class FeaturePerturber(nn.Module):
    """
    Applies perturbation to the features tensor of a VariableLengthFeatures
    object.
    """
    def __init__(
        self,
        batch_size: int,
        input_size: int,
        max_sequence_length: int,
    ):
        """
        :param batch_size: number of samples per batch
        :param input_size: nuber of parameters in input (num cols of features)
        :param max_sequence_length: largest seq length that perturber can
        accommodate (num rows in perturbation matrix)
        """
        super(FeaturePerturber, self).__init__()
        self._batch_size = batch_size
        self.input_size = input_size
        self.max_sequence_length = max_sequence_length
        self.perturbation = nn.Parameter(
            torch.zeros((batch_size, max_sequence_length, input_size))
        )

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def reset_parameters(self):
        """
        Resets perturbation elements to 0
        """
        if self.perturbation.grad is not None:
            self.perturbation.grad.zero_()
        nn.init.zeros_(self.perturbation)

    def forward(
        self, inputs: lms.VariableLengthFeatures
    ) -> lms.VariableLengthFeatures:
        """
        Adds perturbation tensor to features tensor of a VariableLengthFeatures
        object
        :param inputs: a VariableLengthFeatures object
        :return: a new VariableLengthFeatures object (obtained by adding
        perturbation to the .features member of inputs)
        """
        return lms.VariableLengthFeatures(
            features=inputs.features + self.perturbation,
            lengths=inputs.lengths,
        )
