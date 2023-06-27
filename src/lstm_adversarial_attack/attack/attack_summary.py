from functools import cached_property

import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import lstm_adversarial_attack.attack.attack_result_data_structs as ads


class PaddedFeatures:
    features: np.array
    seq_lengths: np.array


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
    def _attacked_features_padded(self) -> np.array:
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

    @property
    def padded_features_success(self) -> np.array:
        return self._attacked_features_padded[
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

    @property
    def first_examples_padded_perts(self) -> np.array:
        return np.array(self._first_examples_raw.perturbations)[
            self._successful_attack_indices
        ]

    @property
    def first_perts_summary(self) -> ads.PertsSummary:
        return ads.PertsSummary(
            seq_lengths=self._seq_lengths[self._successful_attack_indices],
            padded_perts=self.first_examples_padded_perts,
        )

    @property
    def best_examples_padded_perts(self) -> np.array:
        return np.array(self._best_examples_raw.perturbations)[
            self._successful_attack_indices
        ]

    @property
    def best_perts_summary(self) -> ads.PertsSummary:
        return ads.PertsSummary(
            seq_lengths=self._seq_lengths[self._successful_attack_indices],
            padded_perts=self.best_examples_padded_perts,
        )

    @property
    def first_examples_df(self) -> pd.DataFrame:
        data_array = np.stack(
            (
                self._dataset_indices_attacked[
                    self._successful_attack_indices
                ],
                self._orig_labels[self._successful_attack_indices],
                np.array(self._first_examples_raw.epochs)[
                    self._successful_attack_indices
                ],
                np.array(self._first_examples_raw.losses)[
                    self._successful_attack_indices
                ],
                np.array(self._seq_lengths)[self._successful_attack_indices],
                self.first_perts_summary.num_nonzero_elements,
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
                "num_nonzero_perts",
            ],
        )

    @property
    def best_examples_df(self) -> pd.DataFrame:
        data_array = np.stack(
            (
                self._dataset_indices_attacked[
                    self._successful_attack_indices
                ],
                self._orig_labels[self._successful_attack_indices],
                np.array(self._best_examples_raw.epochs)[
                    self._successful_attack_indices
                ],
                np.array(self._best_examples_raw.losses)[
                    self._successful_attack_indices
                ],
                np.array(self._seq_lengths)[self._successful_attack_indices],
                self.best_perts_summary.num_nonzero_elements,
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
                "seq_lengths",
                "num_nonzero_perts",
            ],
        )
