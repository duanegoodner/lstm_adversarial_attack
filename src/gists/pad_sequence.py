import torch
from torch.nn.utils.rnn import pad_sequence

# # assume your input data matrix is stored in a list of tensors
# data = [torch.randn(5, n) for n in [10, 20, 30, 40]]
#
# # pad the sequences to the same length
# padded_data = pad_sequence(data, batch_first=True)
#
# # batch_first=True means that the first dimension of the tensor will be the batch size
#
# print(padded_data.shape)  # prints torch.Size([4, 5, 40])

a = torch.ones(5, 4)
b = torch.ones(7, 4)
c = torch.ones(9, 4)
result = pad_sequence([a, b, c], batch_first=True)
