from pathlib import Path
import torch
import torch.nn as nn


class LSTMSun2018Logit(nn.Module):
    def __init__(
            self,
            model_device: torch.device,
            input_size: int = 48,
            lstm_hidden_size: int = 128,
            fc_hidden_size: int = 32
    ):
        super(LSTMSun2018Logit, self).__init__()
        self.model_device = model_device
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.act_lstm = nn.Tanh()
        # self.dropout = nn.Dropout(p=0.5)
        self.fc_1 = nn.Linear(
            in_features=2 * lstm_hidden_size, out_features=fc_hidden_size
        )
        self.act_1 = nn.ReLU()
        self.fc_2 = nn.Linear(in_features=fc_hidden_size, out_features=2)
        self.to(device=model_device)

    def forward(self, x):
        h_0 = torch.zeros(2, x.size(0), self.lstm_hidden_size).to(
            self.model_device
        )
        c_0 = torch.zeros(2, x.size(0), self.lstm_hidden_size).to(
            self.model_device
        )
        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        lstm_out = self.act_lstm(lstm_out)
        # lstm_out = self.dropout(lstm_out)
        fc_1_out = self.fc_1(lstm_out[:, -1, :])
        fc_1_out = self.act_1(fc_1_out)
        fc_2_out = self.fc_2(fc_1_out)
        return fc_2_out


if __name__ == "__main__":
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    model = LSTMSun2018Logit(model_device=cur_device)
    checkpoint_path = Path(
        "/home/duane/dproj/UIUC-DLH/project/ehr_adversarial_attack/data"
        "/training_results/2023-04-30_18:49:09.556432.tar"
    )
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

