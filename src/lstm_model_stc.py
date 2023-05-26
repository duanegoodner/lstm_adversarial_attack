import torch
import torch.nn as nn
from standard_model_trainer import ModuleWithDevice

class LSTMSun2018(ModuleWithDevice):
    def __init__(
        self,
        device: torch.device,
        input_size: int = 48,
        lstm_hidden_size: int = 128,
        fc_hidden_size: int = 32,
    ):
        super(LSTMSun2018, self).__init__(device=device)
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.act_lstm = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)
        self.fc_1 = nn.Linear(
            in_features=2 * lstm_hidden_size, out_features=fc_hidden_size
        )
        self.act_1 = nn.ReLU()
        self.fc_2 = nn.Linear(in_features=fc_hidden_size, out_features=2)
        self.act_2 = nn.Softmax(dim=1)
        self.to(device=device)

    def forward(self, x: torch.tensor) -> torch.tensor:
        h_0 = torch.zeros(2, x.size(0), self.lstm_hidden_size).to(
            self.device
        )
        c_0 = torch.zeros(2, x.size(0), self.lstm_hidden_size).to(
            self.device
        )
        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        lstm_out = self.act_lstm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        fc_1_out = self.fc_1(lstm_out[:, -1, :])
        fc_1_out = self.act_1(fc_1_out)
        fc_2_out = self.fc_2(fc_1_out)
        out = self.act_2(fc_2_out)
        # out = torch.softmax(fc_2_out, dim=1)
        return out



