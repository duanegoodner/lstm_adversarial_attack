import torch
from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class DatasetWithIndex(ABC, Dataset):

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[int, torch.tensor, torch.tensor]:
        pass



