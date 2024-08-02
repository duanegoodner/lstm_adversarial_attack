import torch
import torch.nn as nn

import lstm_adversarial_attack.attack.attacker_helpers as ah
import lstm_adversarial_attack.attack.feature_perturber as fp


class AdversarialAttacker(nn.Module):
    """
    Provides a forward method to run input features though a perutrber
    and then through a dropout-free logit-out version of a classification model
    """
    def __init__(
        self,
        full_model: nn.Module,
        state_dict: dict,
        input_size: int,
        max_sequence_length: int,
        batch_size: int,
    ):
        """
        :param full_model: original classification mode
        :param state_dict: state dict obtained by previous tuning of model
        :param input_size: number of columns in perturber (matches input
        size of model being attacked)
        :param max_sequence_length: number of rows in perturber (longest
        input sequence that perturber can accept)
        :param batch_size: number of samples per batch
        """
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
        """
        Convenience getter
        """
        return self.feature_perturber.batch_size

    # TODO clean up access to (this class and higher&lower level classes)
    @batch_size.setter
    def batch_size(self, new_batch_size: int):
        self.feature_perturber.batch_size = new_batch_size

    def forward(self, feature: torch.tensor) -> torch.tensor:
        """
        Runs inputs through perturber and then throuh logit-out model
        :param feature: sample input features
        :return: tuple consisting of perturbed feature and logig outputs
        """
        perturbed_feature = self.feature_perturber(feature)
        logits = self.logit_no_dropout_model(perturbed_feature)
        return perturbed_feature, logits

