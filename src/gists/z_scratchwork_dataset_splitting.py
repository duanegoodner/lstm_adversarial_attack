import torch.cuda
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from cv_trainer import CrossValidationTrainer
from lstm_model import BinaryBidirectionalLSTM
from x19_mort_dataset import X19MortalityDataset


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    full_dataset = X19MortalityDataset()

    train_split_size = int(len(full_dataset) * 0.8)
    test_split_size = len(full_dataset) - train_split_size

    train_dataset, test_dataset = random_split(
        full_dataset, (train_split_size, test_split_size)
    )
    model = BinaryBidirectionalLSTM(
        device=device, input_size=48, lstm_hidden_size=128, fc_hidden_size=32
    )
    cv_trainer = CrossValidationTrainer(
        device=device,
        dataset=train_dataset,
        model=model,
        num_folds=5,
        batch_size=128,
        epochs_per_fold=5,
        global_epochs=3,
    )
    cv_trainer.run()


if __name__ == "__main__":
    main()
