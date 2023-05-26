import torch
from pathlib import Path
from torch.utils.data import DataLoader
from cv_trainer import WeightedRandomSamplerBuilder
from lstm_model import BinaryBidirectionalLSTM
from x19_mort_dataset import X19MortalityDataset


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = BinaryBidirectionalLSTM(
        device=device, input_size=48, lstm_hidden_size=128, fc_hidden_size=32
    )

    model_state_path = Path(
        "/data/training_results/2023-04-28-01:52:43.097199.tar"
    )

    checkpoint = torch.load(model_state_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    dataset = X19MortalityDataset()

    data_loader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)

    model.evaluate_model(test_loader=data_loader)
