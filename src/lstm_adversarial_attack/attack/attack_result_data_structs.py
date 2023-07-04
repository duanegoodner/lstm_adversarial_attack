from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from functools import cached_property
from typing import Callable
import lstm_adversarial_attack.dataset_with_index as dsi


@dataclass
class EpochSuccesses:
    epoch_num: int
    batch_indices: torch.tensor
    losses: torch.tensor
    perturbations: torch.tensor


def has_no_entry(loss_vals: torch.tensor, *args, **kwargs) -> torch.tensor:
    return loss_vals == float("inf")


def is_greater_than_new_val(
    loss_vals: torch.tensor, new_loss_vals: torch.tensor
) -> torch.tensor:
    return loss_vals > new_loss_vals.to("cpu")


class RecordedBatchExamples:
    def __init__(
        self,
        batch_size_actual: int,
        max_seq_length: int,
        input_size: int,
        comparison_funct: Callable[..., torch.tensor],
    ):
        self.epochs = torch.empty(batch_size_actual, dtype=torch.long).fill_(
            -1
        )
        self.losses = torch.empty(batch_size_actual).fill_(float("inf"))
        self.perturbations = self.perturbation_first_ex = torch.zeros(
            size=(batch_size_actual, max_seq_length, input_size)
        )
        self.comparison_funct = comparison_funct

    def update(self, epoch_successes: EpochSuccesses):
        loss_values_to_check = self.losses[
            epoch_successes.batch_indices.to("cpu")
        ]

        epoch_indices_to_copy_from = self.comparison_funct(
            loss_values_to_check, epoch_successes.losses
        )

        batch_indices_to_copy_to = epoch_successes.batch_indices[
            epoch_indices_to_copy_from
        ]
        self.epochs[batch_indices_to_copy_to] = epoch_successes.epoch_num
        self.losses[batch_indices_to_copy_to] = epoch_successes.losses[
            epoch_indices_to_copy_from
        ].to("cpu")
        self.perturbations[batch_indices_to_copy_to, :, :] = (
            epoch_successes.perturbations[epoch_indices_to_copy_from, :, :].to(
                "cpu"
            )
        )


class BatchResult:
    def __init__(
        self,
        dataset_indices: torch.tensor,
        input_seq_lengths: torch.tensor,
        max_seq_length: int,
        input_size: int,
    ):
        self.epochs_run = 0
        self.dataset_indices = dataset_indices
        self.input_seq_lengths = input_seq_lengths
        self.first_examples = RecordedBatchExamples(
            batch_size_actual=dataset_indices.shape[0],
            max_seq_length=max_seq_length,
            input_size=input_size,
            comparison_funct=has_no_entry,
        )
        self.best_examples = RecordedBatchExamples(
            batch_size_actual=dataset_indices.shape[0],
            max_seq_length=max_seq_length,
            input_size=input_size,
            comparison_funct=is_greater_than_new_val,
        )

    def update(self, epoch_successes: EpochSuccesses):
        self.epochs_run += 1
        self.first_examples.update(epoch_successes=epoch_successes)
        self.best_examples.update(epoch_successes=epoch_successes)


@dataclass
class RecordedTrainerExamples:
    epochs: torch.tensor = None
    losses: torch.tensor = None
    perturbations: torch.tensor = None
    # device: torch.device = torch.device("cpu")

    def __post_init__(self):
        if self.epochs is None:
            self.epochs = torch.LongTensor()
        if self.losses is None:
            self.losses = torch.FloatTensor()
        if self.perturbations is None:
            self.perturbations = torch.FloatTensor()

    def update(self, batch_examples: RecordedBatchExamples):
        self.epochs = torch.cat((self.epochs, batch_examples.epochs), dim=0)
        self.losses = torch.cat((self.losses, batch_examples.losses), dim=0)
        self.perturbations = torch.cat(
            (self.perturbations, batch_examples.perturbations), dim=0
        )


@dataclass
class TrainerResult:
    dataset: dsi.DatasetWithIndex
    dataset_indices: torch.tensor = None
    epochs_run: torch.tensor = None
    input_seq_lengths: torch.tensor = None
    first_examples: RecordedTrainerExamples = None
    best_examples: RecordedTrainerExamples = None

    def __post_init__(self):
        if self.dataset_indices is None:
            self.dataset_indices = torch.LongTensor()
        if self.epochs_run is None:
            self.epochs_run = torch.LongTensor()
        if self.input_seq_lengths is None:
            self.input_seq_lengths = torch.LongTensor()
        if self.first_examples is None:
            self.first_examples = RecordedTrainerExamples()
        if self.best_examples is None:
            self.best_examples = RecordedTrainerExamples()

    def update(self, batch_result: BatchResult):
        self.first_examples.update(batch_examples=batch_result.first_examples)
        self.best_examples.update(batch_examples=batch_result.best_examples)
        self.dataset_indices = torch.cat(
            (self.dataset_indices, batch_result.dataset_indices)
        )
        self.input_seq_lengths = torch.cat(
            (self.input_seq_lengths, batch_result.input_seq_lengths)
        )
        self.epochs_run = torch.cat(
            (
                self.epochs_run,
                torch.empty(
                    batch_result.input_seq_lengths.shape[0],
                    dtype=torch.long,
                ).fill_(batch_result.epochs_run),
            )
        )

    @property
    def successful_attack(self) -> torch.tensor:
        first_examples_success = np.where(
            self.first_examples.epochs != -1, True, False
        )
        best_examples_success = np.where(
            self.best_examples.epochs != -1, True, False
        )
        assert np.all(first_examples_success == best_examples_success)
        return first_examples_success


class FeaturesMaskBuilder:
    def __init__(self, padded_features_array: np.array, seq_lengths: np.array):
        self._padded_features_array = padded_features_array
        self._seq_lengths = seq_lengths

    def build(self) -> np.array:
        time_indices = np.arange(self._padded_features_array.shape[1])
        time_is_in_range = time_indices.reshape(
            1, -1
        ) < self._seq_lengths.reshape(-1, 1)
        time_is_in_range_bcast = np.broadcast_to(
            time_is_in_range,
            (self._padded_features_array.shape[2], *time_is_in_range.shape),
        )
        return ~np.moveaxis(time_is_in_range_bcast, 0, -1)


class PertsSummary:
    def __init__(
        self,
        seq_lengths: np.array,
        padded_perts: np.array,
    ):
        self._seq_lengths = seq_lengths
        self.padded_perts = padded_perts
        self._mask = FeaturesMaskBuilder(
            padded_features_array=self.padded_perts,
            seq_lengths=self._seq_lengths,
        ).build()
        self.perts = np.ma.array(self.padded_perts, mask=self._mask)

    @cached_property
    def _perts_abs(self) -> np.array:
        return np.abs(self.perts)

    @cached_property
    def _perts_max_positive(self) -> np.array:
        return np.max(self.perts, axis=(1, 2)).data

    @cached_property
    def _perts_max_negative(self) -> np.array:
        return np.min(self.perts, axis=(1, 2)).data

    @cached_property
    def _perts_max_abs(self) -> np.array:
        return np.max(self._perts_abs, axis=(1, 2)).data

    # perts_abs_argmax not needed for now, but leave here a learning reminder
    # https://stackoverflow.com/a/41990983: argmax each sub-array along axis=0
    # @cached_property
    # def perts_abs_argmax(self) -> tuple:
    #     idx = self.perts_abs.reshape(self.perts_abs.shape[0], -1).argmax(axis=-1)
    #     return np.unravel_index(idx, self.perts_abs.shape[-2:])

    @cached_property
    def _perts_abs_sum(self) -> float:
        return np.sum(self._perts_abs.data, axis=(1, 2))

    @cached_property
    def _perts_mean_abs(self) -> np.array:
        return np.mean(self._perts_abs, axis=(1, 2)).data

    @cached_property
    def _perts_mean_nonzero_abs(self) -> np.array:
        return (self._perts_abs_sum / self._num_nonzero_elements).data

    @cached_property
    def _perts_min_nonzero_abs(self) -> np.array:
        zeros_replaced_by_inf = np.where(
            self._perts_abs.data != 0,
            self._perts_abs.data,
            np.inf,
        )
        return np.min(zeros_replaced_by_inf, axis=(1, 2))

    # @cached_property
    # def perts_mean_max_abs(self) -> float:
    #     return np.mean(self.perts_max_abs).item()

    @cached_property
    def _perts_num_actual_elements(self) -> np.array:
        return self._seq_lengths * self.padded_perts.shape[2]

    @cached_property
    def _num_negative_perts(self) -> int:
        return np.sum(self.perts < 0, axis=(1, 2))

    @cached_property
    def _num_positive_perts(self) -> int:
        return np.sum(self.perts > 0, axis=(1, 2))

    @cached_property
    def _num_nonzero_elements(self) -> np.array:
        return self._num_negative_perts + self._num_positive_perts
        # return np.count_nonzero(self.perts_abs.data, axis=(1, 2))

    def num_examples_with_num_nonzero_less_than(self, cutoff: int) -> np.array:
        return np.where(self._num_nonzero_elements < cutoff)[0].shape[0]

    @cached_property
    def _fraction_nonzero(self) -> np.array:
        return self._num_nonzero_elements.astype(
            "float"
        ) / self._perts_num_actual_elements.astype(np.float32)

    @cached_property
    def sparsity(self) -> np.array:
        if len(self._fraction_nonzero) == 0:
            return np.array([], dtype=np.float32)
        else:
            return 1 - self._fraction_nonzero

    @cached_property
    def sparse_small_scores(self) -> np.array:
        if len(self._fraction_nonzero) == 0:
            return np.array([], dtype=np.float32)
        else:
            return (1 - self._fraction_nonzero) / self._perts_abs_sum

    @cached_property
    def sparse_small_max_scores(self) -> np.array:
        if len(self._fraction_nonzero) == 0:
            return np.array([], dtype=np.float32)
        else:
            return self.sparsity / self._perts_max_abs

    @cached_property
    def df(self) -> pd.DataFrame:
        data_array = np.stack(
            (
                self._seq_lengths,
                self._perts_max_positive,
                self._perts_max_negative,
                self._perts_max_abs,
                self._perts_abs_sum,
                self._perts_mean_abs,
                self._perts_mean_nonzero_abs,
                self._perts_min_nonzero_abs,
                self._perts_num_actual_elements,
                self._num_negative_perts,
                self._num_positive_perts,
                self._num_nonzero_elements,
                self._fraction_nonzero,
                self.sparsity,
                self.sparse_small_scores,
                self.sparse_small_max_scores,
            ),
            axis=1,
        )
        return pd.DataFrame(
            data=data_array,
            columns=[
                "seq_length",
                "pert_max_positive",
                "pert_max_negative",
                "pert_max_abs",
                "pert_abs_sum",
                "pert_mean_abs",
                "pert_mean_nonzero_abs",
                "pert_min_nonzero_abs",
                "pert_num_actual_elements",
                "num_negative_perts",
                "num_positive_perts",
                "num_perts",
                "fraction_nonzero",
                "sparsity",
                "sparse_small_scores",
                "sparse_small_max_scores",
            ],
        ).astype(
            dtype={
                "seq_length": "int",
                "pert_max_positive": "float32",
                "pert_max_negative": "float32",
                "pert_max_abs": "float32",
                "pert_abs_sum": "float32",
                "pert_mean_abs": "float32",
                "pert_mean_nonzero_abs": "float32",
                "pert_min_nonzero_abs": "float32",
                "pert_num_actual_elements": "int",
                "num_negative_perts": "int",
                "num_positive_perts": "int",
                "num_perts": "int",
                "fraction_nonzero": "float32",
                "sparsity": "float32",
                "sparse_small_scores": "float32",
                "sparse_small_max_scores": "float32",
            }
        )


class TrainerSuccessSummary:
    def __init__(self, trainer_result: TrainerResult):
        self.dataset = trainer_result.dataset
        self.indices_dataset_attacked = np.array(
            trainer_result.dataset_indices
        )
        self.seq_lengths_attacked = np.array(trainer_result.input_seq_lengths)
        self.epochs_run = trainer_result.epochs_run
        self.successful_attack = trainer_result.successful_attack
        self.first_examples = trainer_result.first_examples
        self.best_examples = trainer_result.best_examples

    def __len__(self):
        return len(self.indices_trainer_success)

    @property
    def indices_trainer_success(self) -> np.array:
        return np.where(self.successful_attack)[0]

    @property
    def indices_dataset_success(self) -> np.array:
        return self.indices_dataset_attacked[self.indices_trainer_success]

    @cached_property
    def orig_labels_attacked(self) -> np.array:
        return np.array(self.dataset[:][2])[self.indices_dataset_attacked]

    @property
    def orig_labels_success(self) -> np.array:
        return np.array(self.dataset[:][2])[self.indices_trainer_success]

    @property
    def seq_lengths_success(self) -> np.array:
        return self.seq_lengths_attacked[self.indices_trainer_success]

    @cached_property
    def all_attacks_df(self) -> pd.DataFrame:
        data_array = np.stack(
            (
                self.indices_dataset_attacked,
                self.orig_labels_attacked,
                self.seq_lengths_attacked,
                self.epochs_run,
                self.successful_attack,
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
        ).astype(
            dtype={
                "dataset_index": "int",
                "orig_label": "int",
                "seq_length": "int",
                "num_epochs_run": "int",
                "successful_attack": "bool",
            }
        )

    @cached_property
    def successful_attacks_df(self) -> pd.DataFrame:
        data_array = np.stack(
            (
                self.indices_dataset_success,
                self.indices_trainer_success,
                self.orig_labels_success,
                self.seq_lengths_success
            ),
            axis=-1
        )
        return pd.DataFrame(
            data=data_array,
            columns=[
                "dataset_index",
                "attacked_samples_index",
                "orig_label",
                "seq_length"
            ]
        )

    @cached_property
    def perts_summary_first(self) -> PertsSummary:
        return PertsSummary(
            seq_lengths=self.seq_lengths_success,
            padded_perts=np.array(
                self.first_examples.perturbations[
                    self.indices_trainer_success, :, :
                ]
            ),
        )

    @cached_property
    def perts_summary_best(self) -> PertsSummary:
        return PertsSummary(
            seq_lengths=self.seq_lengths_success,
            padded_perts=np.array(
                self.best_examples.perturbations[
                    self.indices_trainer_success, :, :
                ]
            ),
        )
