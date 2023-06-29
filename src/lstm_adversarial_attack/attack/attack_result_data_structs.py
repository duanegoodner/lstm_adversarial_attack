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
        self.seq_lengths = seq_lengths
        self.padded_perts = padded_perts
        self.mask = FeaturesMaskBuilder(
            padded_features_array=self.padded_perts,
            seq_lengths=self.seq_lengths,
        ).build()
        self.perts = np.ma.array(self.padded_perts, mask=self.mask)

    @cached_property
    def perts_abs(self) -> np.array:
        return np.abs(self.perts)

    @cached_property
    def perts_max_positive(self) -> np.array:
        return np.max(self.perts, axis=(1, 2)).data

    @cached_property
    def perts_max_negative(self) -> np.array:
        return np.min(self.perts, axis=(1, 2)).data

    @cached_property
    def perts_max_abs(self) -> np.array:
        return np.max(self.perts_abs, axis=(1, 2)).data

    # https://stackoverflow.com/a/41990983: argmax each sub-array along axis=0
    # @cached_property
    # def perts_abs_argmax(self) -> tuple:
    #     idx = self.perts_abs.reshape(self.perts_abs.shape[0], -1).argmax(axis=-1)
    #     return np.unravel_index(idx, self.perts_abs.shape[-2:])

    # @cached_property
    # def perts_max_abs(self) -> np.array:
    #     return np.max(self.perts_abs, axis=(1, 2)).data

    # @cached_property
    # def perts_max_abs(self) -> np.array:
    #     return self.perts_abs[
    #         np.arange(self.perts_abs.shape[0]),
    #         self.perts_abs_argmax[0],
    #         self.perts_abs_argmax[1]
    #     ].data

    # @cached_property
    # def perts_max_signed(self) -> np.array:
    #     return self.perts[
    #         np.arange(self.perts_abs.shape[0]),
    #         self.perts_abs_argmax[0],
    #         self.perts_abs_argmax[1]
    #     ].data
    #
    # @cached_property
    # def perts_max_abs(self) -> np.array:
    #     return np.abs(self.perts_max_signed)

    @cached_property
    def perts_abs_sum(self) -> float:
        return np.sum(self.perts_abs.data, axis=(1, 2))

    @cached_property
    def perts_mean_abs(self) -> np.array:
        return np.mean(self.perts_abs, axis=(1, 2)).data

    @cached_property
    def perts_min_nonzero_abs(self) -> np.array:
        zeros_replaced_by_inf = np.where(
            self.perts_abs.data != 0,
            self.perts_abs.data,
            np.inf,
        )
        return np.min(zeros_replaced_by_inf, axis=(1, 2))

    @cached_property
    def perts_mean_max_abs(self) -> float:
        return np.mean(self.perts_max_abs).item()

    @cached_property
    def perts_num_actual_elements(self) -> np.array:
        return self.seq_lengths * self.padded_perts.shape[2]

    @cached_property
    def num_negative_perts(self) -> int:
        return np.sum(self.perts < 0, axis=(1, 2))

    @cached_property
    def num_positive_perts(self) -> int:
        return np.sum(self.perts > 0, axis=(1, 2))

    @cached_property
    def num_nonzero_elements(self) -> np.array:
        return self.num_negative_perts + self.num_positive_perts
        # return np.count_nonzero(self.perts_abs.data, axis=(1, 2))

    def num_examples_with_num_nonzero_less_than(self, cutoff: int) -> np.array:
        return np.where(self.num_nonzero_elements < cutoff)[0].shape[0]

    @cached_property
    def fraction_nonzero(self) -> np.array:
        return self.num_nonzero_elements.astype(
            "float"
        ) / self.perts_num_actual_elements.astype(np.float32)

    @cached_property
    def sparsity(self) -> np.array:
        if len(self.fraction_nonzero) == 0:
            return np.array([], dtype=np.float32)
        else:
            return 1 - self.fraction_nonzero

    @cached_property
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

