import torch
from torch.utils.data import DataLoader, Dataset
from weighted_dataloader_builder import WeightedDataLoaderBuilder


class MyData(Dataset):
    def __init__(self, x: torch.tensor, y: torch.tensor):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


my_x = torch.arange(0, 20)
my_y = torch.randint(0, 2, (20, ))
my_dataset = MyData(x=my_x, y=my_y)


data_loader_builder = WeightedDataLoaderBuilder()
my_data_loader = data_loader_builder.build(dataset=my_dataset, batch_size=4)

my_data_iterator = iter(my_data_loader)






