# coding: utf-8
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# Sequences
a = torch.tensor([1, 2])
b = torch.tensor([3, 4, 5])
c = torch.tensor([6, 7, 8, 9])
print('a:', a)
print('b:', b)
print('c:', c)

# Settings
seq_lens = [len(a), len(b), len(c)]
max_len = max(seq_lens)


# Zero padding
a = F.pad(a, (0, max_len-len(a)))
b = F.pad(b, (0, max_len-len(b)))
c = F.pad(c, (0, max_len-len(c)))


# Merge the sequences
seq = torch.cat((a, b, c), 0).view(-1, max_len)
print('Sequence:', seq)


# Pack
packed_seq = pack_padded_sequence(seq, seq_lens, batch_first=True, enforce_sorted=False)
print('Pack:', packed_seq)


# Unpack
unpacked_seq, unpacked_lens = pad_packed_sequence(packed_seq, batch_first=True)
print('Unpack:', unpacked_seq)
print('length:', unpacked_lens)


# Reduction
a = unpacked_seq[0][:unpacked_lens[0]]
b = unpacked_seq[1][:unpacked_lens[1]]
c = unpacked_seq[2][:unpacked_lens[2]]
print('Reductions:')
print('a:', a)
print('b:', b)
print('c:', c)