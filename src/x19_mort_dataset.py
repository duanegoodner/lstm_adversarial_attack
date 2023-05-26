import pickle
import torch
from dataset_with_index import DatasetWithIndex
from pathlib import Path
from torch.utils.data import Dataset
import project_config as pc


def get_pickle_data(file: Path):
    with file.open(mode="rb") as pickle_file:
        return pickle.load(pickle_file)


class X19MortalityDataset(Dataset):
    def __init__(
            self,
            x19_pickle: Path = pc.GOOD_PICKLE_DIR / pc.X19_PICKLE_FILENAME,
            y_pickle: Path = pc.GOOD_PICKLE_DIR / pc.Y_PICKLE_FILENAME
    ):
        orig_x19 = torch.from_numpy(get_pickle_data(x19_pickle)).float()
        self.x19 = torch.permute(orig_x19, (0, 2, 1))

        # y pickle file has many targets per sample. we want mort (1st entry)
        all_y = get_pickle_data(y_pickle)
        self.mort = torch.tensor([icu_stay[0] for icu_stay in all_y])

    def __len__(self):
        return len(self.mort)

    def __getitem__(self, idx: int):
        return self.x19[idx, :, :], self.mort[idx]


class X19MortalityDatasetWithIndex(X19MortalityDataset, DatasetWithIndex):

    def __getitem__(self, idx: int):
        return idx, self.x19[idx, :, :], self.mort[idx]


if __name__ == "__main__":
    my_dataset = X19MortalityDataset()
