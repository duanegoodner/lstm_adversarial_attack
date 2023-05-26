# https://chat.openai.com/c/86de3751-5731-4bc4-964d-dd416c83108c


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


# define the bidirectional LSTM model
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2,
                            num_classes)  # *2 for bidirectional LSTM

    def forward(self, x, seq_lengths):
        # sort input sequences by length in descending order
        seq_lengths, sort_idx = seq_lengths.sort(0, descending=True)
        x = x[sort_idx]

        # pack padded sequences
        packed = nn.utils.rnn.pack_padded_sequence(x,
                                                   seq_lengths.cpu().numpy(),
                                                   batch_first=True)

        # pass through LSTM layers
        out, _ = self.lstm(packed)

        # unpack padded sequences
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # get the last output of each sequence
        idx = (seq_lengths - 1).view(-1, 1).expand(len(seq_lengths),
                                                   out.size(2))
        out = out.gather(1, idx.unsqueeze(1)).squeeze(1)

        # pass through linear layer and return output
        out = self.fc(out)
        return out


# generate sample input data
my_data = [torch.randn(n, 5) for n in [10, 20, 30, 40]]

# pad the sequences to the same length
my_padded_data = pad_sequence(my_data, batch_first=True)

# generate the sequence lengths tensor
my_seq_lengths = torch.LongTensor([d.shape[0] for d in my_data])

# define the model with input_size=5, hidden_size=10, num_layers=2, and num_classes=2
my_model = BiLSTM(input_size=5, hidden_size=10, num_layers=2, num_classes=2)

# forward pass through the model
my_output = my_model(my_padded_data, my_seq_lengths)

print(my_output.shape)  # prints torch.Size([4, 2])
