import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from IPython.display import HTML, display
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).parent.parent))
import lstm_adversarial_attack.config as config
# import lstm_adversarial_attack.config_paths as cfp
import lstm_adversarial_attack.preprocess.encode_decode as edc
from lstm_adversarial_attack.data_structures import VariableLengthFeatures
from lstm_adversarial_attack.dataset_with_index import DatasetWithIndex


class X19MGeneralDataset(Dataset):
    def __init__(
        self,
        measurements: list[torch.tensor],
        in_hosp_mort: list[torch.tensor],
        max_num_samples: int = None,
        random_seed: int = None,
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

    def __len__(self):
        return len(self.in_hosp_mort)

    def __getitem__(self, idx: int):
        return self.measurements[idx], self.in_hosp_mort[idx]

    @classmethod
    # TODO change this to import from json files instead of pickles (will use
    #  preprocess.encode_decode.FeatureArraysReader & .ClassLabelsReader
    def from_feature_finalizer_output(
        cls,
        # measurements_path: Path = cfp.PREPROCESS_OUTPUT_DIR
        # / cfp.PREPROCESS_OUTPUT_FILES["measurement_data_list"],
        # in_hospital_mort_path: Path = cfp.PREPROCESS_OUTPUT_DIR
        # / cfp.PREPROCESS_OUTPUT_FILES["in_hospital_mortality_list"],
        max_num_samples: int = None,
        random_seed: int = None,
    ):
        config_reader = config.ConfigReader()
        measurements_path = Path(
            config_reader.read_path(
                config_key="dataset.resources.measurement_data_list"
            )
        )
        in_hospital_mort_path = Path(
            config_reader.read_path(
                config_key="dataset.resources.in_hospital_mortality_list"
            )
        )
        measurements_np_list = (
            edc.FeatureArraysReader().import_struct(path=measurements_path).data
        )
        mort_int_list = (
            edc.ClassLabelsReader().import_struct(path=in_hospital_mort_path).data
        )
        assert len(measurements_np_list) == len(mort_int_list)

        features_tensor_list = [
            torch.tensor(entry, dtype=torch.float32) for entry in measurements_np_list
        ]
        labels_tensor_list = [
            torch.tensor(entry, dtype=torch.long) for entry in mort_int_list
        ]
        return cls(
            measurements=features_tensor_list,
            in_hosp_mort=labels_tensor_list,
            max_num_samples=max_num_samples,
            random_seed=random_seed,
        )


def x19m_collate_fn(batch):
    features, labels = zip(*batch)
    padded_features = pad_sequence(sequences=features, batch_first=True)
    lengths = torch.tensor([item.shape[0] for item in features], dtype=torch.long)
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
    lengths = torch.tensor([item.shape[0] for item in features], dtype=torch.long)
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
        dataset_entry_type = type(self.dataset[0]).__name__
        dataset_entry_length = len(self.dataset[0])
        input_feature_data_struct = type(self.dataset[0][0]).__name__
        input_feature_dims = self.dataset[0][0].dim()
        input_size = self.dataset[0][0].shape[1]
        input_feature_dtype = self.dataset[0][0].dtype
        label_data_struct = type(self.dataset[0][1]).__name__
        label_dims = self.dataset[0][1].dim()
        label_dtype = self.dataset[0][1].dtype

        summary = (
            f"There are {dataset_size} samples in the Dataset.\nCalling"
            f" `__getitem__` on the Dataset returns a {dataset_entry_type} of"
            f" length {dataset_entry_length}.\nThe first element of this tuple"
            f" is a {input_feature_dims}-D {input_feature_data_struct} with"
            f" {input_size} columns and data type {input_feature_dtype}.\nThe"
            f" second element is a {label_dims}-D {label_data_struct} with"
            f" data type {label_dtype}"
        )

        print(summary)

    def view_seq_length_summary(self):
        unique_sequence_lengths, sequence_length_counts = np.unique(
            [item.shape[0] for item in self.dataset[:][0]],
            return_counts=True,
        )

        summary_df = pd.DataFrame(
            data=np.stack((unique_sequence_lengths, sequence_length_counts), axis=1),
            columns=["seq_length", "num_samples"],
        )

        summary_df.set_index("seq_length", inplace=True)
        display(HTML(summary_df.T.to_html()))

    def view_label_summary(self):
        unique_labels, label_counts = np.unique(
            [self.dataset[:][1]], return_counts=True
        )
        summary_df = pd.DataFrame(
            data=np.stack((unique_labels, label_counts), axis=1),
            columns=["class_label", "nunm_samples"],
        )

        summary_df.set_index("class_label", inplace=True)
        display(HTML(summary_df.T.to_html()))


if __name__ == "__main__":
    x19_general_dataset = X19MGeneralDataset.from_feature_finalizer_output(
        # max_num_samples=100
    )
    x19_with_index = X19MGeneralDatasetWithIndex.from_feature_finalizer_output(
        # max_num_samples=100
    )
