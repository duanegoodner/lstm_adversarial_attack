import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class SimpleDataset(Dataset):
    def __init__(
        self, features: np.array, targets: np.array
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.tensor, torch.tensor]:
        return self.features[idx], self.targets[idx]


class FlexibleAdder(nn.Module):
    def __init__(self, device: torch.device, adder_dims: tuple[int, int]):
        super(FlexibleAdder, self).__init__()
        self._device = device
        self._adder_dims = adder_dims
        self.amount_to_add = nn.Parameter(0.01 * torch.randn(*adder_dims))
        self.to(self._device)
        self.reset_parameters()

    def reset_parameters(self):
        if self.amount_to_add.grad is not None:
            self.amount_to_add.grad.zero_()
        self.amount_to_add.data = torch.rand_like(self.amount_to_add)

    def forward(self, x: torch.tensor) -> torch.tensor:
        out = x + self.amount_to_add
        return out


class AdderTrainer:
    def __init__(
        self, device: torch.device, adder: FlexibleAdder, dataset: Dataset
    ):
        self._device = device
        self._adder = adder
        self._optimizer = torch.optim.SGD(params=adder.parameters(), lr=0.05)
        self._loss_fn = torch.nn.MSELoss()
        self._dataset = dataset

    def build_single_sample_data_loader(self) -> DataLoader:
        return DataLoader(dataset=self._dataset, batch_size=1)

    def train_adder(self, num_epochs: int):
        dataloader = self.build_single_sample_data_loader()

        self._adder.train()
        all_best_x = torch.FloatTensor()
        all_best_y_pred = torch.FloatTensor()
        for num_batches, (x, y) in enumerate(dataloader):
            self._adder.reset_parameters()
            lowest_loss = torch.inf
            best_x = torch.zeros_like(x)
            best_y_pred = torch.zeros_like(y)
            x, y = x.to(self._device), y.to(self._device)
            for epoch in range(num_epochs):
                self._optimizer.zero_grad()
                y_pred = self._adder(x)
                loss = self._loss_fn(y_pred, y)
                if loss.item() < lowest_loss:
                    lowest_loss = loss.item()
                    best_x = self._adder.amount_to_add.detach().to("cpu")
                    best_y_pred = y_pred.detach().to("cpu")
                loss.backward()
                self._optimizer.step()
            all_best_x = torch.cat((all_best_x, best_x))
            all_best_y_pred = torch.cat((all_best_y_pred, best_y_pred))
        return all_best_x, all_best_y_pred


if __name__ == "__main__":
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    simple_features = np.array([
        np.full(shape=(2, 2), fill_value=0.5),
        np.full(shape=(2, 2), fill_value=0.3)
    ])
    simple_targets = np.array([
        np.full(shape=(2, 2), fill_value=1.0, dtype=np.single),
        np.full(shape=(2, 2), fill_value=1.0, dtype=np.single)
    ])

    dataset = SimpleDataset(features=simple_features, targets=simple_targets)
    adder = FlexibleAdder(device=cur_device, adder_dims=(2, 2))
    trainer = AdderTrainer(device=cur_device, adder=adder, dataset=dataset)

    result = trainer.train_adder(num_epochs=50)







