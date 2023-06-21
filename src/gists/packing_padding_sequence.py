# https://gist.github.com/xssChauhan/57b0d8db83b832fbc13253a8bc0c156e
import torch

# Generate Input 1, with labels for each element
a = torch.ones(10)
a_labels = torch.ones(10)

# Generate Input 2, with labels for each element
b = torch.ones(5)
b_labels = torch.ones(5)

# Pad and pack the sequences so that PyTorch does not waste time on computation for the paddings
a_b = torch.nn.utils.rnn.pad_sequence([a,b], batch_first=True)
a_b = torch.nn.utils.rnn.pack_padded_sequence(a_b, lengths=[10,5], batch_first=True)

# Pack the labels so that the input is consistent in shape
a_b_labels_1 = torch.nn.utils.rnn.pad_sequence([a_labels, b_labels], batch_first=True)
a_b_labels = torch.nn.utils.rnn.pack_padded_sequence(a_b_labels_1, lengths=[10,5], batch_first=True)

# Assume the inputs have been passed through some LSTM/RNN layer

# Unpack, or pad the packed labels
c = torch.nn.utils.rnn.pad_packed_sequence(a_b, batch_first=True)
c_labels = torch.nn.utils.rnn.pad_packed_sequence(a_b_labels, batch_first=True)


# See that the labels are the same
print(c_labels, a_b_labels_1)