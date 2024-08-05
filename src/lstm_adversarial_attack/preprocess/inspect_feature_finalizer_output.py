import argparse
import sys
import pandas as pd
from functools import cached_property
from pathlib import Path
from typing import List

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.preprocess.encode_decode as edc
import lstm_adversarial_attack.utils.path_searches as ps
from lstm_adversarial_attack.config import CONFIG_READER, PATH_CONFIG_READER


class FeatureFinalizerOutputInspector:
    def __init__(self, preprocess_id: str):
        self.preprocess_id = preprocess_id
        self.preprocess_data_root = Path(
            PATH_CONFIG_READER.read_path("preprocess.output_root")
        )

    @cached_property
    def features(self) -> List[np.ndarray]:
        features_path = (
            self.preprocess_data_root
            / self.preprocess_id
            / PATH_CONFIG_READER.get_config_value(
                "dataset.resources.measurement_data_list"
            )["preprocess"]
        )
        return edc.FeatureArraysReader().import_struct(path=features_path).data

    @cached_property
    def class_labels(self) -> list[int]:
        class_labels_path = (
            self.preprocess_data_root
            / self.preprocess_id
            / PATH_CONFIG_READER.get_config_value(
                "dataset.resources.in_hospital_mortality_list"
            )["preprocess"]
        )
        return (
            edc.ClassLabelsReader().import_struct(path=class_labels_path).data
        )

    @property
    def num_samples(self) -> int:
        return len(self.features)

    @property
    def sequence_length_distribution(self):
        return np.unique(
            [item.shape[0] for item in self.features], return_counts=True
        )

    @property
    def num_measurements_distribution(self):
        return np.unique(
            [item.shape[1] for item in self.features], return_counts=True
        )

    @property
    def class_labels_distribution(self):
        return np.unique(self.class_labels, return_counts=True)

    def display_class_labels_distribution(self):
        unique_class_labels, label_counts = self.class_labels_distribution
        summary_df = pd.DataFrame(
            data=np.stack([unique_class_labels, label_counts], axis=1),
            columns=["class_label", "count"],
        )
        summary_df.set_index("class_label", inplace=True)
        print(summary_df.T)

    @staticmethod
    def display_distribution(
        distribution_property: tuple, parameter_name: str, title: str
    ):
        unique_values, counts = distribution_property
        summary_df = pd.DataFrame(
            data=np.stack([unique_values, counts], axis=1),
            columns=[parameter_name, "count"],
        )
        summary_df.set_index(parameter_name, inplace=True)
        print(title)
        print("-" * len(title))
        print(summary_df.T)
        print()


def main(preprocess_id: str = None):

    if preprocess_id is None:
        preprocess_output_root = Path(
            PATH_CONFIG_READER.read_path("preprocess.output_root")
        )
        preprocess_id = ps.get_latest_sequential_child_dirname(
            root_dir=preprocess_output_root
        )

    output_inspector = FeatureFinalizerOutputInspector(
        preprocess_id=preprocess_id
    )

    print(f"Summary of output from preprocess session {preprocess_id}\n")

    print(f"Number of samples = {output_inspector.num_samples}\n")

    output_inspector.display_distribution(
        distribution_property=output_inspector.sequence_length_distribution,
        parameter_name="sequence_length",
        title="Time Series Sequence Length Distribution",
    )

    output_inspector.display_distribution(
        distribution_property=output_inspector.num_measurements_distribution,
        parameter_name="num_measurements",
        title="Measurement Column Counts Distribution",
    )

    print(
        f"Min value of any element in any feature matrix = {np.min(output_inspector.features)}"
    )
    print(
        f"Max value of any element in any feature matrix = {np.max(output_inspector.features)}\n"
    )

    output_inspector.display_distribution(
        distribution_property=output_inspector.class_labels_distribution,
        parameter_name="class_label",
        title="Class Labels Distribution",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--preprocess_id",
        type=str,
        action="store",
        nargs="?",
        help="ID of preprocess session to use as data source. Defaults to latest session",
    )
    args_namespace = parser.parse_args()
    main(preprocess_id=args_namespace.preprocess_id)
