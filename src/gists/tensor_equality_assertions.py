import torch


a = torch.tensor([1, 2, 3])
b = torch.tensor([1, 2, 3])

assert (a == b).all()