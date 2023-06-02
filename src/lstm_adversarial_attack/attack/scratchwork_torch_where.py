import torch

# Original tensor
original_tensor = torch.tensor([
    [[1.0, 2.0],
     [3.0, 4.0]],

    [[5.0, 6.0],
     [7.0, 8.0]],

    [[9.0, 10.0],
     [11.0, 12.0]]
])

# Boolean tensor
bool_tensor = torch.tensor([True, False, True])

# Create a mask
mask = bool_tensor.view(3, 1, 1)

# Apply the mask to the original tensor
result_tensor = torch.where(mask, original_tensor[2], original_tensor[0])

# Print the result tensor
print(result_tensor)
