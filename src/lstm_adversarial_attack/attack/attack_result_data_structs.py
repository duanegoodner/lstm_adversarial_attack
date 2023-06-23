import numpy as np
import torch
from dataclasses import dataclass
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
    device: torch.device = torch.device("cpu")

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


class PerturbationSummary:
    def __init__(
        self, padded_perts: torch.tensor, input_seq_lengths: torch.tensor
    ):
        self.padded_perts = padded_perts
        self.input_seq_lengths = input_seq_lengths
        self.actual_perts = [
            padded_perts[i, : input_seq_lengths[i], :]
            for i in range(input_seq_lengths.shape[0])
        ]
        self.abs_perts = [torch.abs(item) for item in self.actual_perts]

        self.sum_abs_perts = torch.tensor(
            [torch.sum(item) for item in self.abs_perts]
        )

        self.mean_pert_magnitudes = torch.tensor(
            [torch.mean(item) for item in self.abs_perts]
        )
        self.stdev_pert_magnitudes = torch.tensor(
            [torch.std(item) for item in self.abs_perts]
        )
        self.min_pert_magnitudes = torch.tensor(
            [torch.min(item) for item in self.abs_perts]
        )
        self.max_pert_magnitudes = torch.tensor(
            [torch.max(item) for item in self.abs_perts]
        )
        self.mean_of_means = torch.mean(self.mean_pert_magnitudes)
        self.num_actual_elements = input_seq_lengths * padded_perts.shape[2]
        self.num_nonzero = torch.tensor(
            [torch.count_nonzero(item) for item in self.actual_perts]
        )
        self.fraction_nonzero = (
            self.num_nonzero.float() / self.num_actual_elements.float()
        )

        if len(self.fraction_nonzero) == 0:
            self.sparsity = torch.tensor([], dtype=torch.float32)
        else:
            self.sparsity = 1 - self.fraction_nonzero

        if len(self.fraction_nonzero) == 0:
            self.sparse_small_scores = torch.tensor([], dtype=torch.float32)
        else:
            self.sparse_small_scores = (
                1 - self.fraction_nonzero
            ) / self.sum_abs_perts

        self.fraction_nonzero_mean = torch.mean(self.fraction_nonzero)
        self.fraction_nonzero_stdev = torch.std(self.fraction_nonzero)
        if len(self.fraction_nonzero) == 0:
            self.fraction_nonzero_min = torch.tensor([], dtype=torch.float32)
        else:
            self.fraction_nonzero_min = torch.min(self.fraction_nonzero)

    # @property
    # def actual_perts(self) -> list[torch.tensor]:


class TrainerSuccessSummary:
    def __init__(self, trainer_result: TrainerResult):
        best_success_trainer_indices = torch.where(
            trainer_result.best_examples.epochs != -1
        )[0]
        first_success_trainer_indices = torch.where(
            trainer_result.first_examples.epochs != -1
        )[0]
        assert (
            (best_success_trainer_indices == first_success_trainer_indices)
            .all()
            .item()
        )

        self.dataset = trainer_result.dataset
        self.attacked_dataset_indices = trainer_result.dataset_indices
        self.success_dataset_indices = trainer_result.dataset_indices[
            best_success_trainer_indices
        ]
        self.epochs_run = trainer_result.epochs_run[
            best_success_trainer_indices
        ]
        self.input_seq_lengths = trainer_result.input_seq_lengths[
            best_success_trainer_indices
        ]
        self.first_examples = RecordedTrainerExamples(
            epochs=trainer_result.first_examples.epochs[
                first_success_trainer_indices
            ],
            losses=trainer_result.first_examples.losses[
                first_success_trainer_indices
            ],
            perturbations=trainer_result.first_examples.perturbations[
                first_success_trainer_indices, :, :
            ],
        )
        self.best_examples = RecordedTrainerExamples(
            epochs=trainer_result.best_examples.epochs[
                best_success_trainer_indices
            ],
            losses=trainer_result.best_examples.losses[
                best_success_trainer_indices
            ],
            perturbations=trainer_result.best_examples.perturbations[
                best_success_trainer_indices, :, :
            ],
        )
        self.first_perts_summary = PerturbationSummary(
            padded_perts=self.first_examples.perturbations,
            input_seq_lengths=self.input_seq_lengths,
        )
        self.best_perts_summary = PerturbationSummary(
            padded_perts=self.best_examples.perturbations,
            input_seq_lengths=self.input_seq_lengths,
        )

    def get_filtered_perts(
        self,
        perts_type: str = None,
        seq_length: int = None,
        orig_label: int = None,
    ) -> torch.tensor:
        assert perts_type == "first" or perts_type == "best"
        full_examples = (
            self.first_examples
            if perts_type == "first"
            else self.best_examples
        )

        if seq_length is not None:
            match_seq_length_summary_indices = torch.where(
                self.input_seq_lengths == seq_length
            )[0]
        else:
            match_seq_length_summary_indices = torch.arange(
                len(self.input_seq_lengths)
            )

        label_tensor = torch.tensor(self.dataset[:][2])
        success_orig_labels = label_tensor[self.success_dataset_indices]
        if orig_label is not None:
            match_label_dataset_indices = torch.where(
                success_orig_labels == orig_label
            )[0]
        else:
            match_label_dataset_indices = torch.arange(
                len(self.success_dataset_indices)
            )

        filtered_indices = np.intersect1d(
            match_seq_length_summary_indices, match_label_dataset_indices
        )

        filtered_perts = full_examples.perturbations[filtered_indices, :, :]
        filtered_seq_lengths = self.input_seq_lengths[filtered_indices]

        return filtered_perts
