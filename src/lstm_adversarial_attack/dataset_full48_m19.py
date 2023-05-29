import numpy as np
import torch
from dataset_with_index import DatasetWithIndex
from pathlib import Path
from torch.utils.data import Dataset
import preprocess.preprocess_input_classes as pic
import project_config_old as pc
import resource_io as rio


class Full48M19Dataset(Dataset):
    """
    Intended for datasets with same number of observation hours (typically
    48) for all samples. Best way to produce this data is to use
    FeatureFinalizer with settings.max_hours = 48 &
    settings.require_exact_num_hours = True.
    """

    def __init__(self, measurements: torch.tensor, in_hosp_mort: torch.tensor):
        self.measurements = measurements
        self.in_hosp_mort = in_hosp_mort

    def __len__(self):
        return len(self.in_hosp_mort)

    def __getitem__(self, idx: int):
        return self.measurements[idx, :, :], self.in_hosp_mort[idx]

    @classmethod
    def from_feature_finalizer_output(
        cls,
        measurements_path: Path = pc.PREPROCESS_OUTPUT_DIR
        / pc.PREPROCESS_OUTPUT_FILES["measurement_data_list"],
        in_hospital_mort_path: Path = pc.PREPROCESS_OUTPUT_DIR
        / pc.PREPROCESS_OUTPUT_FILES["in_hospital_mortality_list"],
    ):
        """
        Creates Full48M19Dataset using paths to pickles created by
        FeatureFinalizer. Function is (overly) complicated b/c
        FeatureFinalizer could return feature arrays of more than single
        length, and b/c dataset needs to provide same format as
        X19MortalityDataset.
        """
        importer = rio.ResourceImporter()
        measurements_list = importer.import_pickle_to_object(
            path=measurements_path
        )
        in_hosp_mort_list = importer.import_pickle_to_object(
            path=in_hospital_mort_path
        )
        unique_num_hours = np.unique(
            [item.shape[0] for item in measurements_list]
        )
        assert len(unique_num_hours) == 1
        assert len(in_hosp_mort_list) == len(measurements_list)

        measurements_np = np.array(measurements_list, dtype=np.float32)
        in_hosp_mort_np = np.array(in_hosp_mort_list, dtype=np.int64)

        measurements_tensor = torch.from_numpy(measurements_np)
        # measurements_tensor = torch.permute(measurements_tensor, (0, 2, 1))
        in_hosp_mort_tensor = torch.from_numpy(in_hosp_mort_np)

        return cls(
            measurements=measurements_tensor, in_hosp_mort=in_hosp_mort_tensor
        )


class Full48M19DatasetWithIndex(Full48M19Dataset, DatasetWithIndex):
    def __getitem__(self, idx: int):
        return idx, self.measurements[idx, :, :], self.in_hosp_mort[idx]


if __name__ == "__main__":
    full48_m19_dataset = Full48M19Dataset.from_feature_finalizer_output()
