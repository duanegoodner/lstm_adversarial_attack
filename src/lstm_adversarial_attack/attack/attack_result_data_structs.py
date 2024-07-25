from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Callable

import numpy as np
import pandas as pd
import torch

import lstm_adversarial_attack.dataset_with_index as dsi


@dataclass
class EpochSuccesses:
    """
    Container for results from one epoch of attacks on a batch of samples
    :param epoch_num: number/index of epoch
    :param batch_indices: indices (within batch) of successful attacks
    :param losses: loss vals of successful attacks
    :param perturbations: perturbation tensors resulting in successful attacks
    """
    epoch_num: int
    batch_indices: torch.tensor
    losses: torch.tensor
    perturbations: torch.tensor


def has_no_entry(loss_vals: torch.tensor, *args, **kwargs) -> torch.tensor:
    """
    Helper function to determine which elements of RecordedBatchExample do
    not correspond to a found adversarial example
    :param loss_vals: tensor of loss vals
    :return: boolean tensor with true for entries == float("inf")
    """

    return loss_vals == float("inf")


def is_greater_than_new_val(
    loss_vals: torch.tensor, new_loss_vals: torch.tensor
) -> torch.tensor:
    """
    Helper function for comparing tensors of values
    :param loss_vals: loss vals that were previously stored
    :param new_loss_vals: loss vals from latest epoch
    :return: tensor of bools
    """
    return loss_vals > new_loss_vals.to("cpu")


class RecordedBatchExamples:
    """
    Stores results of attacks of on samples in batch. Will have one
    RecordedBatchExamples object for first found example, and one for best (
    i.e. lowest loss) example.
    """
    def __init__(
        self,
        batch_size_actual: int,
        max_seq_length: int,
        input_size: int,
        comparison_funct: Callable[..., torch.tensor],
    ):
        """

        :param batch_size_actual: number of samples actually in batch (
        only time may differ from regular batch_size is on final batch)
        :param max_seq_length: longest input sequence length (num rows)
        :param input_size: size of input tensor for sample (num cols)
        :param comparison_funct: function for comparing example_data (and
        deciding which gets saved as a "best" example)
        """
        self.epochs = torch.empty(batch_size_actual, dtype=torch.long).fill_(
            -1
        )
        self.losses = torch.empty(batch_size_actual).fill_(float("inf"))
        self.perturbations = self.perturbation_first_ex = torch.zeros(
            size=(batch_size_actual, max_seq_length, input_size)
        )
        self.comparison_funct = comparison_funct

    def update(self, epoch_successes: EpochSuccesses):
        """
        Updates self with info from an EpochSuccesses object
        :param epoch_successes: EpochSuccesses (of latest epoch)
        :return:
        """
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
    """
    Container for RecordedBatchExamples corresponding to first and best
    adversarial example_data of each sample in batch.
    """
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
        """
        Updated both RecordedBatchExamples members with info from latest
        EpochSuccesses
        :param epoch_successes: result of latest epoch
        """
        self.epochs_run += 1
        self.first_examples.update(epoch_successes=epoch_successes)
        self.best_examples.update(epoch_successes=epoch_successes)


@dataclass
class RecordedTrainerExamples:
    """
    Compilation of RecordedBatchExamples from all batches in dataset
    :param epochs: tensor of ints indicating epoch when example was found
    :param losses: tensor of loss vals from each example
    :param perturbations: 3d tensor of perturbations (batch dim = 0)
    """
    epochs: torch.tensor = None
    losses: torch.tensor = None
    perturbations: torch.tensor = None

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
    """
    Compilation of RecordedTrainerExamples from each batch
    :param dataset: dataset being attacked
    :param dataset_indices: dataset indices of samples being attacked
    :param epochs_run: num epochs run on each batch (should all be same val)
    :param input_seq_lengths: input seq length of each sample
    :param first_examples: info from first found example for samples
    :param best_examples: info for best (lowest loss) example_data
    """
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
        """
        Updates self with info from a BatchResult
        :param batch_result: latest BatchResult
        """
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
        """
        Boolean tensor indicating whether sample had at least one successful
        attack
        :return: tensor of bools
        """
        first_examples_success = np.where(
            self.first_examples.epochs != -1, True, False
        )
        best_examples_success = np.where(
            self.best_examples.epochs != -1, True, False
        )
        assert np.all(first_examples_success == best_examples_success)
        return first_examples_success


class FeaturesMaskBuilder:
    """
    Builds boolean mask to mask out rows in features array that do not
    correspond to input (for samples with seq length < max input length)
    """
    def __init__(self, padded_features_array: np.array, seq_lengths: np.array):
        """
        :param padded_features_array: padded 3d array of input features
        :param seq_lengths: actual seq length of each sample
        """
        self._padded_features_array = padded_features_array
        self._seq_lengths = seq_lengths

    def build(self) -> np.array:
        """
        Creates boolean mask w/ value = False if corresponding element in
        padded features array does not correspond to actual input
        :return: array of bools
        """
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
    """
    Performs summary calculations using perturbations from first and best
    adversarial example_data found by attack on a datast.
    """
    def __init__(
        self,
        seq_lengths: np.array,
        padded_perts: np.array,
    ):
        """
        :param seq_lengths: array of input seq lengths
        :param padded_perts: 3d arrray of padded perturbations. Batch = axis 0.
        """
        self._seq_lengths = seq_lengths
        self.padded_perts = padded_perts
        self._mask = FeaturesMaskBuilder(
            padded_features_array=self.padded_perts,
            seq_lengths=self._seq_lengths,
        ).build()

        # will use a masked array for most calculations
        self.perts = np.ma.array(self.padded_perts, mask=self._mask)

    @cached_property
    def _perts_abs(self) -> np.ma.MaskedArray:
        """
        Gets abs value of masked perts array.
        :return: masked array of floats
        """
        return np.abs(self.perts)

    @cached_property
    def _perts_max_positive(self) -> np.array:
        """
        Largest positive element of each 2d subarray of perts along 0 axis
        (one for each sample with an example in perts summary)
        :return: array of floats
        """
        return np.max(self.perts, axis=(1, 2)).data

    @cached_property
    def _perts_max_negative(self) -> np.array:
        """
        Largest magnitude negative element of each 2d subarray of perts along 0 axis
        (one for each sample with an example in perts summary)
        :return: array of floats
        """
        return np.min(self.perts, axis=(1, 2)).data

    @cached_property
    def _perts_max_abs(self) -> np.array:
        """
        Largest magnitude perturbation element
        :return: array of floats
        """
        return np.max(self._perts_abs, axis=(1, 2)).data

    # perts_abs_argmax not needed for now, but leave here a learning reminder
    # https://stackoverflow.com/a/41990983: argmax each sub-array along axis=0
    # @cached_property
    # def perts_abs_argmax(self) -> tuple:
    #     idx = self.perts_abs.reshape(self.perts_abs.shape[0], -1).argmax(axis=-1)
    #     return np.unravel_index(idx, self.perts_abs.shape[-2:])

    @cached_property
    def _perts_abs_sum(self) -> np.array:
        """
        Sum of perturbation matrix magnitude for each sample
        :return: array of floats
        """
        return np.sum(self._perts_abs.data, axis=(1, 2))

    @cached_property
    def _perts_mean_abs(self) -> np.array:
        """
        Average perturbation element magnitude of each sample
        :return: array of floats
        """
        return np.mean(self._perts_abs, axis=(1, 2)).data

    @cached_property
    def _perts_mean_nonzero_abs(self) -> np.array:
        """
        Average mean magnitude of nonzero perturbation elements for each
        sample.
        :return: array of floats
        """
        return (self._perts_abs_sum / self._num_nonzero_elements).data

    @cached_property
    def _perts_min_nonzero_abs(self) -> np.array:
        """
        Smallest magnitude of any nonzero pert element for each sample
        :return: array of floats
        """
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
        """
        Number of actual feature inputs for each sample
        :return: array of ints
        """
        return self._seq_lengths * self.padded_perts.shape[2]

    @cached_property
    def _num_negative_perts(self) -> np.ma.MaskedArray:
        """
        Number of negative perturbation elements for each sample
        :return: masked array of ints
        """
        return np.sum(self.perts < 0, axis=(1, 2))

    @cached_property
    def _num_positive_perts(self) -> np.ma.MaskedArray:
        """
        Number of positive perturbation elements for each sample
        :return: masked array of ints
        """
        return np.sum(self.perts > 0, axis=(1, 2))

    @cached_property
    def _num_nonzero_elements(self) -> np.ma.MaskedArray:
        """
        Number of nonzero perturbation elements for each sample
        :return: masked array of ints
        """
        return self._num_negative_perts + self._num_positive_perts
        # return np.count_nonzero(self.perts_abs.data, axis=(1, 2))

    def num_examples_with_num_nonzero_less_than(self, cutoff: int) -> np.array:
        """
        Gets number of example_data with number of nonzero perts below cutoff val.
        :param cutoff: example_data must have less than this num nonzero elements
        :return: array of ints
        """
        return np.where(self._num_nonzero_elements < cutoff)[0].shape[0]

    @cached_property
    def _fraction_nonzero(self) -> np.ma.MaskedArray:
        """
        Fraction of perturbation elements that are nonzero (excluding padding
        vals)
        :return: masked array of floats
        """
        return self._num_nonzero_elements.astype(
            "float"
        ) / self._perts_num_actual_elements.astype(np.float32)

    @cached_property
    def sparsity(self) -> np.ma.MaskedArray:
        """
        Sparsity of each example's perturbation
        :return: masked array of floats
        """
        if len(self._fraction_nonzero) == 0:
            return np.array([], dtype=np.float32)
        else:
            return 1 - self._fraction_nonzero

    @cached_property
    def sparse_small_scores(self) -> np.ma.MaskedArray:
        """
        Sparse-small score of each example
        :return: masked array of floats
        """
        if len(self._fraction_nonzero) == 0:
            return np.array([], dtype=np.float32)
        else:
            return (1 - self._fraction_nonzero) / self._perts_abs_sum

    @cached_property
    def sparse_small_max_scores(self) -> np.ma.MaskedArray:
        """
        sparse-small-max score of each example
        :return: masked array of floats
        """
        if len(self._fraction_nonzero) == 0:
            return np.array([], dtype=np.float32)
        else:
            return self.sparsity / self._perts_max_abs

    @cached_property
    def df(self) -> pd.DataFrame:
        """
        Gets a dataframe summarizing calculated values for each example
        :return: Pandas Dataframe
        """
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
    """
    Summarizes first and best example data from an attack
    """
    def __init__(self, trainer_result: TrainerResult):
        """
        :param trainer_result: TrainerResult object from attack on dataset
        """
        self.dataset = trainer_result.dataset
        self.indices_dataset_attacked = np.array(
            trainer_result.dataset_indices
        )
        self.seq_lengths_attacked = np.array(trainer_result.input_seq_lengths)
        self.epochs_run = trainer_result.epochs_run
        self.successful_attack = trainer_result.successful_attack
        self.first_examples = trainer_result.first_examples
        self.best_examples = trainer_result.best_examples

    def __len__(self) -> int:
        """
        Number of samples with found example
        :return: int
        """
        return len(self.indices_trainer_success)

    @property
    def indices_trainer_success(self) -> np.array:
        """
        Gets indices withing array of attacked samples of samples with
        successful attack
        :return: array of ints
        """
        return np.where(self.successful_attack)[0]

    @property
    def indices_dataset_success(self) -> np.array:
        """
        Gets dataset indices of samples with successful attacks
        :return: array of ints
        """
        return self.indices_dataset_attacked[self.indices_trainer_success]

    @cached_property
    def orig_labels_attacked(self) -> np.array:
        """
        Class of each attacked sample predicted by target model
        :return: array of ints
        """
        return np.array(self.dataset[:][2])[self.indices_dataset_attacked]

    @property
    def orig_labels_success(self) -> np.array:
        """
        Predicted class (by model) of each successfully attacked sample
        :return: array of ints
        """
        return np.array(self.dataset[:][2])[self.indices_trainer_success]

    @property
    def seq_lengths_success(self) -> np.array:
        """
        Input seq length of each successfully attacked sample
        :return: array of ints
        """
        return self.seq_lengths_attacked[self.indices_trainer_success]

    @cached_property
    def all_attacks_df(self) -> pd.DataFrame:
        """
        Dataframe summarizing success/fail of all attacked samples
        :return: Pandas Dataframe
        """
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
        """
        Dataframe summarizing attack successes (first and best)
        :return: Pandas Dataframe
        """
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
        """
        PertsSummary of first found example_data for each sample that was
        successfully attacked
        :return: a PertsSummary object
        """
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
        """
        PertsSummary of best found example_data for each sample that was
        successfully attacked
        :return: a PertsSummary object
        """
        return PertsSummary(
            seq_lengths=self.seq_lengths_success,
            padded_perts=np.array(
                self.best_examples.perturbations[
                    self.indices_trainer_success, :, :
                ]
            ),
        )
