import torch
import torch.nn as nn


my_model_a = nn.LSTM(
    input_size=20, hidden_size=128, bidirectional=True, batch_first=True
)

if torch.cuda.is_available():
    cur_device = torch.device("cuda:0")
else:
    cur_device = torch.device("cpu")

my_model_a = my_model_a.to(cur_device)

my_model_a = my_model_a.to("cpu")




