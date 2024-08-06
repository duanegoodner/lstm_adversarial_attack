import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).parent.parent))
import lstm_adversarial_attack.preprocess.encode_decode as edc
from lstm_adversarial_attack.model.lstm_model_stc import VariableLengthFeatures
from lstm_adversarial_attack.dataset.dataset_with_index import DatasetWithIndex
from lstm_adversarial_attack.config.read_write import PATH_CONFIG_READER


@dataclass
class X19MGeneralDatasetInfo:
    preprocess_id: str
    max_num_samples: int
    random_seed: int


class X19MGeneralDataset(Dataset):
    def __init__(
        self,
        measurements: list[torch.tensor],
        in_hosp_mort: list[torch.tensor],
        max_num_samples: int = None,
        random_seed: int = None,
        preprocess_id: str = None,
    ):
        if max_num_samples is not None and max_num_samples < len(in_hosp_mort):
            if random_seed:
                np.random.seed(random_seed)
            selected_indices = np.random.choice(
                a=len(in_hosp_mort), size=max_num_samples, replace=False
            )
            measurements = [measurements[i] for i in selected_indices]
            in_hosp_mort = [in_hosp_mort[i] for i in selected_indices]

        self.measurements = measurements
        self.in_hosp_mort = in_hosp_mort
        self.preprocess_id = preprocess_id

    def __len__(self):
        return len(self.in_hosp_mort)

    def __getitem__(self, idx: int):
        return self.measurements[idx], self.in_hosp_mort[idx]

    @classmethod
    def from_feature_finalizer_output(
        cls,
        preprocess_id: str,
        max_num_samples: int = None,
        random_seed: int = None,
    ):

        preprocess_data_root = Path(
            PATH_CONFIG_READER.read_path("preprocess.output_root")
        )

        measurements_path = (
            preprocess_data_root
            / preprocess_id
            / PATH_CONFIG_READER.get_value(
                "dataset.resources.measurement_data_list"
            )["preprocess"]
        )

        in_hospital_mort_path = (
            preprocess_data_root
            / preprocess_id
            / PATH_CONFIG_READER.get_value(
                "dataset.resources.in_hospital_mortality_list"
            )["preprocess"]
        )
        measurements_np_list = (
            edc.FeatureArraysReader()
            .import_struct(path=measurements_path)
            .data
        )
        mort_int_list = (
            edc.ClassLabelsReader()
            .import_struct(path=in_hospital_mort_path)
            .data
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
            measurements=features_tensor_list,
            in_hosp_mort=labels_tensor_list,
            max_num_samples=max_num_samples,
            random_seed=random_seed,
            preprocess_id=preprocess_id,
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


class X19MGeneralDatasetWithIndex(X19MGeneralDataset, DatasetWithIndex):
    def __getitem__(self, idx: int):
        return idx, self.measurements[idx], self.in_hosp_mort[idx]


def x19m_with_index_collate_fn(batch):
    indices, features, labels = zip(*batch)
    padded_features = pad_sequence(sequences=features, batch_first=True)
    lengths = torch.tensor(
        [item.shape[0] for item in features], dtype=torch.long
    )
    return (
        torch.tensor(indices, dtype=torch.long),
        VariableLengthFeatures(features=padded_features, lengths=lengths),
        torch.tensor(labels, dtype=torch.long),
    )


class DatasetInspector:
    """
    Provides methods for displaying basic dataset info.
    Intended for use in a Jupyter notebook
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def view_basic_info(self):
        dataset_size = len(self.dataset)
        min__number_feature_rows = min([item[0].shape[0] for item in self.dataset])
        max_number_feature_rows = max([item[0].shape[0] for item in self.dataset])
        min__number_feature_cols = min([item[0].shape[1] for item in self.dataset])
        max_number_feature_cols = max([item[0].shape[1] for item in self.dataset])
        unique_class_label_vals = np.unique([item[1] for item in self.dataset])

        print(f"Number of samples = {dataset_size}\n")
        print("Size ranges of input feature tensors:")
        print(f"Min # rows = {min__number_feature_rows}")
        print(f"Max # rows = {max_number_feature_rows}")
        print(f"Min # columns = {min__number_feature_cols}")
        print(f"Max # columns = {max_number_feature_cols}\n")
        print(f"Unique class label vals = {unique_class_label_vals}\n")


    def view_seq_length_summary(self):
        unique_sequence_lengths, sequence_length_counts = np.unique(
            [item.shape[0] for item in self.dataset[:][0]],
            return_counts=True,
        )

        summary_df = pd.DataFrame(
            data=np.stack(
                (unique_sequence_lengths, sequence_length_counts), axis=1
            ),
            columns=["seq_length", "num_samples"],
        )

        summary_df.set_index("seq_length", inplace=True)
        print("Sequence Length Distribution")
        print(summary_df.T)

    def view_label_summary(self):
        unique_labels, label_counts = np.unique(
            [self.dataset[:][1]], return_counts=True
        )
        summary_df = pd.DataFrame(
            data=np.stack((unique_labels, label_counts), axis=1),
            columns=["class_label", "num_samples"],
        )

        summary_df.set_index("class_label", inplace=True)
        print("Class Label Distribution")
        print(summary_df.T)


if __name__ == "__main__":
    x19_general_dataset = X19MGeneralDataset.from_feature_finalizer_output(
        # max_num_samples=100
    )
    x19_with_index = X19MGeneralDatasetWithIndex.from_feature_finalizer_output(
        # max_num_samples=100
    )
