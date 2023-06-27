from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from enum import Enum, auto
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


class RecordedExampleType(Enum):
    FIRST = auto()
    BEST = auto()


class PertsSummary:
    def __init__(
        self,
        seq_lengths: np.array,
        padded_perts: np.array,
    ):
        self.seq_lengths = seq_lengths
        self.padded_perts = padded_perts
        self.mask = self.create_mask()
        self.perts = np.ma.array(self.padded_perts, mask=self.mask)

    def create_filtered_perts_summary(
        self, indices_to_keep: np.array
    ) -> PertsSummary:
        return PertsSummary(
            seq_lengths=self.seq_lengths[indices_to_keep],
            padded_perts=self.padded_perts[indices_to_keep, :, :],
        )

    def create_mask(self) -> np.array:
        time_indices = np.arange(self.padded_perts.shape[1])
        time_is_in_range = time_indices.reshape(
            1, -1
        ) < self.seq_lengths.reshape(-1, 1)
        time_is_in_range_bcast = np.broadcast_to(
            time_is_in_range,
            (self.padded_perts.shape[2], *time_is_in_range.shape),
        )
        return ~np.moveaxis(time_is_in_range_bcast, 0, -1)

    @cached_property
    def perts_abs(self) -> np.array:
        return np.abs(self.perts)

    @property
    def perts_abs_sum(self) -> float:
        return np.sum(self.perts_abs.data, axis=(1, 2))

    @property
    def perts_mean_abs(self) -> np.array:
        return np.mean(self.perts_abs, axis=(1, 2)).data

    @property
    def perts_min_nonzero_abs(self) -> np.array:
        zeros_replaced_by_inf = np.where(
            self.perts_abs.data != 0,
            self.perts_abs.data,
            np.inf,
        )
        return np.min(zeros_replaced_by_inf, axis=(1, 2))

    @property
    def perts_max_abs(self) -> np.array:
        return np.max(self.perts_abs, axis=(1, 2)).data

    @property
    def perts_mean_max_abs(self) -> float:
        return np.mean(self.perts_max_abs).item()

    @property
    def perts_num_actual_elements(self) -> np.array:
        return self.seq_lengths * self.padded_perts.shape[2]

    @property
    def num_nonzero_elements(self) -> np.array:
        return np.count_nonzero(self.perts_abs.data, axis=(1, 2))

    def num_examples_with_num_nonzero_less_than(self, cutoff: int) -> np.array:
        return np.where(self.num_nonzero_elements < cutoff)[0].shape[0]

    @property
    def fraction_nonzero(self) -> np.array:
        return self.num_nonzero_elements.astype(
            "float"
        ) / self.perts_num_actual_elements.astype(np.float32)

    @property
    def sparsity(self) -> np.array:
        if len(self.fraction_nonzero) == 0:
            return np.array([], dtype=np.float32)
        else:
            return 1 - self.fraction_nonzero

    @property
    def sparse_small_scores(self) -> np.array:
        if len(self.fraction_nonzero) == 0:
            return np.array([], dtype=torch.float32)
        else:
            return (1 - self.fraction_nonzero) / self.perts_abs_sum


class TrainerSuccessSummary:
    def __init__(self, trainer_result: TrainerResult):
        self.trainer_result = trainer_result

    def __len__(self):
        return len(self.indices_trainer_success)

    @property
    def indices_dataset_attacked(self) -> np.array:
        return np.array(self.trainer_result.dataset_indices)

    @property
    def indices_trainer_success(self) -> np.array:
        first_indices_success_trainer = np.where(
            self.trainer_result.first_examples.epochs != -1
        )[0]

        best_indices_success_trainer = np.where(
            self.trainer_result.best_examples.epochs != -1
        )[0]

        assert np.all(
            first_indices_success_trainer == best_indices_success_trainer
        )
        return best_indices_success_trainer

    @property
    def indices_dataset_success(self) -> np.array:
        return self.indices_dataset_attacked[self.indices_trainer_success]

    @cached_property
    def orig_labels_attacked(self) -> np.array:
        return np.array(self.trainer_result.dataset[:][2])[
            self.indices_dataset_attacked
        ]

    @property
    def orig_labels_success(self) -> np.array:
        return np.array(self.trainer_result.dataset[:][2])[
            self.indices_dataset_success
        ]

    @property
    def seq_lengths_attacked(self) -> np.array:
        return np.array(self.trainer_result.input_seq_lengths)

    @property
    def seq_lengths_success(self) -> np.array:
        return self.seq_lengths_attacked[self.indices_trainer_success]

    @cached_property
    def examples_summary_first(self) -> PertsSummary:
        return PertsSummary(
            seq_lengths=self.seq_lengths_success,
            padded_perts=np.array(
                self.trainer_result.first_examples.perturbations[
                    self.indices_trainer_success, :, :
                ]
            ),
        )

    @cached_property
    def examples_summary_best(self) -> PertsSummary:
        return PertsSummary(
            seq_lengths=self.seq_lengths_success,
            padded_perts=np.array(
                self.trainer_result.best_examples.perturbations[
                    self.indices_trainer_success, :, :
                ]
            ),
        )

    def filtered_examples_summary(
        self,
        recorded_example_type: RecordedExampleType,
        seq_length_min: int = None,
        seq_length_max: int = None,
        orig_label: int = None,
        min_num_nonzero_perts: int = None,
        max_num_nonzero_perts: int = None,
    ) -> PertsSummary:
        dispatch = {
            RecordedExampleType.FIRST: self.examples_summary_first,
            RecordedExampleType.BEST: self.examples_summary_best,
        }

        success_idx_filtered_by_min_seq_length = (
            np.where(self.seq_lengths_success >= seq_length_min)[0]
            if seq_length_min is not None
            else np.arange(len(self))
        )

        success_idx_filtered_by_max_seq_length = (
            np.where(self.seq_lengths_success <= seq_length_max)[0]
            if seq_length_max is not None
            else np.arange(len(self))
        )

        success_idx_length_filtered = np.intersect1d(
            ar1=success_idx_filtered_by_min_seq_length,
            ar2=success_idx_filtered_by_max_seq_length,
        )

        success_idx_filtered_orig_label = (
            np.where(self.orig_labels_success == orig_label)
            if orig_label is not None
            else np.arange(len(self))
        )

        success_idx_len_label_filtered = np.intersect1d(
            ar1=success_idx_length_filtered,
            ar2=success_idx_filtered_orig_label,
        )

        full_summary = dispatch[recorded_example_type]

        success_idx_min_num_nonzero_filtered = np.where(
            full_summary.num_nonzero_elements >= min_num_nonzero_perts
            if min_num_nonzero_perts is not None
            else np.arange(len(self))
        )

        success_idx_max_num_nonzero_filtered = np.where(
            full_summary.num_nonzero_elements <= max_num_nonzero_perts
            if max_num_nonzero_perts is not None
            else np.arange(len(self))
        )

        success_idx_num_nonzero_filtered = np.intersect1d(
            ar1=success_idx_min_num_nonzero_filtered,
            ar2=success_idx_max_num_nonzero_filtered
        )

        success_idx_filtered = np.intersect1d(
            ar1=success_idx_len_label_filtered,
            ar2=success_idx_num_nonzero_filtered,
        )

        return full_summary.create_filtered_perts_summary(
            indices_to_keep=success_idx_filtered
        )

