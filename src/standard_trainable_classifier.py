import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class StandardTrainableClassifier(nn.Module, ABC):
    def __init__(
        self,
        model_device: torch.device,
    ):
        super(StandardTrainableClassifier, self).__init__()
        self.model_device = model_device
        self.to(device=model_device)

    @abstractmethod
    def forward(self, x: torch.tensor) -> torch.tensor:
        pass
