import sys
import torch
import torch.nn as nn
from pathlib import Path
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
)

sys.path.append(str(Path(__file__).parent.parent))
from lstm_adversarial_attack.data_structures import VariableLengthFeatures


# use this as component in nn.Sequential of full model
class BidirectionalLSTMX19(nn.Module):
    def __init__(
        self,
        input_size: int = 19,
        lstm_hidden_size: int = 128,
    ):
        super(BidirectionalLSTMX19, self).__init__()
        self.input_size = input_size
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


class BidirectionalLSTMX19ForTensorboard(nn.Module):
    """
    This variant of BidirectionalLSTMX19 takes a list of tensors (padded to
    same size along axis 1) as inputs instead of a list of
    VariableLengthFeatures. Intended for use with
    SummaryWriter.add_graph(model, inputs) method which can only handle tensor
    or list of tensors (of equal length) for its input arg.
    """

    def __init__(self, input_size: int = 19, lstm_hidden_size: int = 128):
        super(BidirectionalLSTMX19ForTensorboard, self).__init__()
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, inputs: torch.tensor) -> torch.tensor:
        # padded_inputs = pad_sequence(inputs, batch_first=True)
        lstm_out, (h_n, c_n) = self.lstm(inputs)
        return lstm_out
