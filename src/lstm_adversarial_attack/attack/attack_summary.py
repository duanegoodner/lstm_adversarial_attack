from enum import Enum, auto
from functools import cached_property

import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import lstm_adversarial_attack.attack.attack_result_data_structs as ads


class RecordedExampleType(Enum):
    FIRST = auto()
    BEST = auto()


class AttackResults:
    def __init__(self, trainer_result: ads.TrainerResult):
        self._dataset = trainer_result.dataset
        self._dataset_indices_attacked = np.array(
            trainer_result.dataset_indices
        )
        self._epochs_run = np.array(trainer_result.epochs_run)
        self._seq_lengths = np.array(trainer_result.input_seq_lengths)
        self._first_examples_raw = trainer_result.first_examples
        self._best_examples_raw = trainer_result.best_examples

    @cached_property
    def attacked_features_padded(self) -> np.array:
        torch_attacked_features_padded = pad_sequence(
            self._dataset[:][1], batch_first=True
        )[self._dataset_indices_attacked]
        return np.array(torch_attacked_features_padded)

    @cached_property
    def _successful_attack(self) -> np.array:
        first_success_attack_indices = np.where(
            self._first_examples_raw.epochs != -1, True, False
        )
        best_success_attack_indices = np.where(
            self._best_examples_raw.epochs != -1, True, False
        )
        assert np.all(
            first_success_attack_indices == best_success_attack_indices
        )
        return first_success_attack_indices

    @cached_property
    def _successful_attack_indices(self) -> np.array:
        return np.where(self._successful_attack)[0]

    @cached_property
    def padded_features(self) -> np.array:
        torch_padded_features = pad_sequence(
            self._dataset[:][1], batch_first=True
        )
        return np.array(torch_padded_features)

    @cached_property
    def padded_features_success(self) -> np.array:
        return self.attacked_features_padded[
            self._successful_attack_indices, :, :
        ]

    @cached_property
    def _orig_labels(self) -> np.array:
        return np.array(self._dataset[:][2])[self._dataset_indices_attacked]

    @cached_property
    def attacked_samples_df(self) -> pd.DataFrame:
        data_array = np.stack(
            (
                self._dataset_indices_attacked,
                self._orig_labels,
                self._seq_lengths,
                self._epochs_run,
                self._successful_attack,
            ),
            axis=-1,
        )
        return pd.DataFrame(
            data=data_array,
            columns=[
                "dataset_index",
                "orig_label",
                "seq_length",
                "num_epochs_run",
                "successful_attack",
            ],
        )

    @cached_property
    def first_examples_padded_perts(self) -> np.array:
        return np.array(self._first_examples_raw.perturbations)[
            self._successful_attack_indices
        ]

    @cached_property
    def first_perts_summary(self) -> ads.PertsSummary:
        return ads.PertsSummary(
            seq_lengths=self._seq_lengths[self._successful_attack_indices],
            padded_perts=self.first_examples_padded_perts,
        )

    @cached_property
    def best_examples_padded_perts(self) -> np.array:
        return np.array(self._best_examples_raw.perturbations)[
            self._successful_attack_indices
        ]

    @cached_property
    def best_perts_summary(self) -> ads.PertsSummary:
        return ads.PertsSummary(
            seq_lengths=self._seq_lengths[self._successful_attack_indices],
            padded_perts=self.best_examples_padded_perts,
        )

    def build_examples_summary_df(
        self,
        example_type: RecordedExampleType,
    ):
        examples_dispatch = {
            RecordedExampleType.FIRST: self._first_examples_raw,
            RecordedExampleType.BEST: self._best_examples_raw,
        }

        perts_summary_dispatch = {
            RecordedExampleType.FIRST: self.first_perts_summary,
            RecordedExampleType.BEST: self.best_perts_summary,
        }

        trainer_examples = examples_dispatch[example_type]
        perts_summary = perts_summary_dispatch[example_type]

        data_array = np.stack(
            (
                self._dataset_indices_attacked[
                    self._successful_attack_indices
                ],
                self._orig_labels[self._successful_attack_indices],
                np.array(trainer_examples.epochs)[
                    self._successful_attack_indices
                ],
                np.array(trainer_examples.losses)[
                    self._successful_attack_indices
                ],
                np.array(self._seq_lengths)[self._successful_attack_indices],
                perts_summary.num_nonzero_elements,
                perts_summary.num_negative_perts,
                perts_summary.perts_max_negative,
                perts_summary.num_positive_perts,
                perts_summary.perts_max_positive
            ),
            axis=1,
        )
        return pd.DataFrame(
            data=data_array,
            columns=[
                "dataset_index",
                "orig_label",
                "epoch_found",
                "loss",
                "seq_length",
                "num_perts",
                "num_negative_perts",
                "pert_max_negative",
                "num_positive_perts",
                "pert_max_positive"
            ]
        ).astype(dtype={
                "dataset_index": "int",
                "orig_label": "int",
                "epoch_found" : "int",
                "loss": "float32",
                "seq_length": "int",
                "num_perts": "int",
                "num_negative_perts": "int",
                "pert_max_negative": "float32",
                "num_positive_perts": "int",
                "pert_max_positive": "float32"
            })

    @cached_property
    def first_examples_df(self) -> pd.DataFrame:
        return self.build_examples_summary_df(
            example_type=RecordedExampleType.FIRST
        )

    @cached_property
    def best_examples_df(self) -> pd.DataFrame:
        return self.build_examples_summary_df(
            example_type=RecordedExampleType.BEST
        )
