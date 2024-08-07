import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
from pathlib import Path
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)
sys.path.append(str(Path(__file__).parent.parent))
# from lstm_adversarial_attack.model.model_data_structs import VariableLengthFeatures
from lstm_adversarial_attack.config.read_write import CONFIG_READER


@dataclass
class VariableLengthFeatures:
    features: torch.tensor
    lengths: torch.tensor

class BidirectionalX19LSTM(nn.Module):
    """
    Bidirectional, single layer LSTM that takes VariableLengthFeatures objects
    as input.
    """
    def __init__(
        self,
        lstm_hidden_size: int,
    ):
        super(BidirectionalX19LSTM, self).__init__()

        bg_data_cols = CONFIG_READER.get_value("preprocess.bg_data_cols")
        lab_data_cols = CONFIG_READER.get_value("preprocess.lab_data_cols")
        vital_data_cols = CONFIG_READER.get_value("preprocess.vital_data_cols")
        input_size = len(bg_data_cols) + len(lab_data_cols) + len(vital_data_cols)

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            bidirectional=True,
            batch_first=True,
        )

    def forward(
        self,
        variable_length_features: VariableLengthFeatures,
    ) -> torch.tensor:
        """
        Converts batch of VariableLengthFeatures to PackedSequence, runs
        through LSTM, then unpacks. .lengths of inputs used when packing and
        unpacking.
        :param variable_length_features: object with two data members - tensor
        of batch_size x max_input_seq_length x input_size (num measurements),
        and integer specifying actual seq_length.
        :return: unpacked LSTM outputs
        """
        packed_features = pack_padded_sequence(
            variable_length_features.features,
            lengths=variable_length_features.lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        lstm_out_packed, (h_n, c_n) = self.lstm(packed_features)
        unpacked_lstm_out, lstm_out_lengths = pad_packed_sequence(
            sequence=lstm_out_packed, batch_first=True
        )
        final_lstm_out = unpacked_lstm_out[
            torch.arange(unpacked_lstm_out.shape[0]), lstm_out_lengths - 1, :
        ].squeeze()
        return final_lstm_out


class BidirectionalLSTMX19Graph(nn.Module):
    """
    Not recommended for modeling use. Only for model graph visualizations that
    require only feature tensor input.
    """
    def __init__(self, input_size: int = 19, lstm_hidden_size: int = 128):
        super(BidirectionalLSTMX19Graph, self).__init__()
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, inputs: torch.tensor) -> torch.tensor:
        lstm_out, (h_n, c_n) = self.lstm(inputs)
        return lstm_out
