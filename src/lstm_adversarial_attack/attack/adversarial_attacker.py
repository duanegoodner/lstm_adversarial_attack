import torch.nn as nn


class AdversarialAttacker:
    def __init__(self, trained_model: nn.Module):