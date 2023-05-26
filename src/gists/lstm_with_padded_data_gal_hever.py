# https://galhever.medium.com/sentiment-analysis-with-pytorch-part-4-lstm-bilstm-model-84447f6c4525


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable


class LSTM(nn.Module):
    # define all the layers used in model
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        lstm_units,
        hidden_dim,
        num_classes,
        lstm_layers,
        bidirectional,
        dropout,
        pad_index,
        batch_size,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_index
        )
        self.lstm = nn.LSTM(
            embedding_dim,
            lstm_units,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        num_directions = 2 if bidirectional else 1
        self.fc1 = nn.Linear(lstm_units * num_directions, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm_layers = lstm_layers
        self.num_directions = num_directions
        self.lstm_units = lstm_units

    def init_hidden(self, batch_size):
        h, c = (
            Variable(
                torch.zeros(
                    self.lstm_layers * self.num_directions,
                    batch_size,
                    self.lstm_units,
                )
            ),
            Variable(
                torch.zeros(
                    self.lstm_layers * self.num_directions,
                    batch_size,
                    self.lstm_units,
                )
            ),
        )
        return h, c

    def forward(self, text, text_lengths):
        batch_size = text.shape[0]
        h_0, c_0 = self.init_hidden(batch_size)

        embedded = self.embedding(text)
        packed_embedded = pack_padded_sequence(
            embedded, text_lengths, batch_first=True
        )
        output, (h_n, c_n) = self.lstm(packed_embedded, (h_0, c_0))
        output_unpacked, output_lengths = pad_packed_sequence(
            output, batch_first=True
        )
        out = output_unpacked[:, -1, :]
        rel = self.relu(out)
        dense1 = self.fc1(rel)
        drop = self.dropout(dense1)
        preds = self.fc2(drop)
        return preds
