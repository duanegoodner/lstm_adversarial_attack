import torch
import torch.nn as nn


class SingleSampleFeaturePerturber(nn.Module):
    def __init__(
        self,
        device: torch.device,
        feature_dims: tuple[int, ...] | torch.Size,
    ):
        super(SingleSampleFeaturePerturber, self).__init__()
        self._device = device
        self.perturbation = nn.Parameter(
            torch.zeros(feature_dims, dtype=torch.float32)
        )
        self.to(self._device)

    def reset_parameters(self):
        if self.perturbation.grad is not None:
            self.perturbation.grad.zero_()
        nn.init.zeros_(self.perturbation)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return x + self.perturbation
