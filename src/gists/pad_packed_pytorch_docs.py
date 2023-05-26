import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
    pack_sequence,
)

a = torch.tensor([[1, 1], [2, 2]], dtype=torch.float32)
b = torch.tensor([[3, 3]], dtype=torch.float32)
c = torch.tensor([[4, 4], [5, 5], [6, 6]], dtype=torch.float32)
d = torch.tensor([[7, 7], [8, 8], [9, 9], [10, 10]], dtype=torch.float32)

padded_sequences = pad_sequence(sequences=[a, b, c, d], batch_first=True)
lengths = [item.shape[0] for item in [a, b, c, d]]
packed_input = pack_padded_sequence(
    input=padded_sequences,
    lengths=lengths,
    batch_first=True,
    enforce_sorted=False,
)

lstm = nn.LSTM(input_size=2, hidden_size=5)

packed_output, (hidden, cell) = lstm(packed_input)
unpacked_output, lengths = pad_packed_sequence(
    sequence=packed_output, batch_first=True
)


# unpack input for comparing w/ output
seq_unpacked, lens_unpacked = pad_packed_sequence(
    packed_input, batch_first=True
)


# packed_input_v2 = pack_sequence([a, b, c, d], enforce_sorted=False)
# seq_unpacked_v2, lens_unpacked_v2 = pad_packed_sequence(
#     packed_input_v2, batch_first=True
# )
