import torch

# logits = torch.randn((5, 2))
# labels = torch.randint(low=0, high=2, size=(logits.shape[0],))

import torch

# Create the 3 x 2 tensor
tensor_3x2 = torch.tensor([[1.0, 2.0],
                           [3.0, 4.0],
                           [5.0, 6.0]])

# Create the 1-D tensor
tensor_1d = torch.tensor([0, 1, 0])

# Use the 1-D tensor as indices to select columns from the 3 x 2 tensor
new_tensor_1d = tensor_3x2[torch.arange(tensor_3x2.size(0)), tensor_1d]

print(new_tensor_1d)