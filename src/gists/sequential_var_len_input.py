import torch
import torch.nn as nn



my_model = nn.Sequential(
    nn.LSTM(input_size=5, hidden_size=10),
    nn.ReLU()
)

for model_idx, model in enumerate(my_model.modules()):
    print(model_idx)

