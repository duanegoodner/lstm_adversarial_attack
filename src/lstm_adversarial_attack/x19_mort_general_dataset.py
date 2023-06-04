import sys
import torch
from lstm_adversarial_attack.dataset_with_index import DatasetWithIndex
from pathlib import Path
from torch.nn.utils.rnn import (
    pad_sequence,
)
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).parent.parent))
from lstm_adversarial_attack.config_paths import PREPROCESS_OUTPUT_DIR, PREPROCESS_OUTPUT_FILES
from lstm_adversarial_attack.data_structures import VariableLengthFeatures

# import project_config_old as pc
import lstm_adversarial_attack.resource_io as rio


class X19MGeneralDataset(Dataset):
    def __init__(
        self, measurements: list[torch.tensor], in_hosp_mort: torch.tensor
    ):
        self.measurements = measurements
        self.in_hosp_mort = in_hosp_mort

    def __len__(self):
        return len(self.in_hosp_mort)

    def __getitem__(self, idx: int):
        return self.measurements[idx], self.in_hosp_mort[idx]

    @classmethod
    def from_feaure_finalizer_output(
        cls,
        measurements_path: Path = PREPROCESS_OUTPUT_DIR
        / PREPROCESS_OUTPUT_FILES["measurement_data_list"],
        in_hospital_mort_path: Path = PREPROCESS_OUTPUT_DIR
        / PREPROCESS_OUTPUT_FILES["in_hospital_mortality_list"],
    ):
        importer = rio.ResourceImporter()
        measurements_np_list = importer.import_pickle_to_object(
            path=measurements_path
        )
        mort_int_list = importer.import_pickle_to_object(
            path=in_hospital_mort_path
        )
        assert len(measurements_np_list) == len(mort_int_list)

        features_tensor_list = [
            torch.tensor(entry, dtype=torch.float32)
            for entry in measurements_np_list
        ]
        labels_tensor_list = [
            torch.tensor(entry, dtype=torch.long) for entry in mort_int_list
        ]
        return cls(
            measurements=features_tensor_list, in_hosp_mort=labels_tensor_list
        )


def x19m_collate_fn(batch):
    features, labels = zip(*batch)
    padded_features = pad_sequence(sequences=features, batch_first=True)
    lengths = torch.tensor(
        [item.shape[0] for item in features], dtype=torch.long
    )
    return VariableLengthFeatures(
        features=padded_features, lengths=lengths
    ), torch.tensor(labels, dtype=torch.long)
    # return padded_features, torch.tensor(labels, dtype=torch.long), lengths


class X19MGeneralDatasetWithIndex(X19MGeneralDataset, DatasetWithIndex):
    def __getitem__(self, idx: int):
        return idx, self.measurements[idx], self.in_hosp_mort[idx]


def x19m_with_index_collate_fn(batch):
    indices, features, labels = zip(*batch)
    padded_features = pad_sequence(sequences=features, batch_first=True)
    lengths = torch.tensor(
        [item.shape[0] for item in features], dtype=torch.long
    )
    return torch.tensor(indices, dtype=torch.long), VariableLengthFeatures(
        features=padded_features, lengths=lengths
    ), torch.tensor(labels, dtype=torch.long)


if __name__ == "__main__":
    x19_general_dataset = X19MGeneralDataset.from_feaure_finalizer_output()
    x19_with_index = X19MGeneralDatasetWithIndex.from_feaure_finalizer_output()