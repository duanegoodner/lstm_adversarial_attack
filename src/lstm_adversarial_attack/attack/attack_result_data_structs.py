import numpy as np
import torch
from dataclasses import dataclass
from typing import Callable


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
    return loss_vals > new_loss_vals


class RecordedBatchExamples:
    def __init__(
        self,
        initial_device: torch.device,
        batch_size_actual: int,
        max_seq_length: int,
        input_size: int,
        comparison_funct: Callable[..., torch.tensor],
    ):
        self.epochs = torch.empty(batch_size_actual, dtype=torch.long).fill_(
            -1
        ).to(initial_device)
        self.losses = torch.empty(batch_size_actual).fill_(float("inf")).to(initial_device)
        self.perturbations = self.perturbation_first_ex = torch.zeros(
            size=(batch_size_actual, max_seq_length, input_size)
        ).to(initial_device)
        self.comparison_funct = comparison_funct

    def to(self, device: torch.device):
        self.epochs = self.epochs.to(device)
        self.losses = self.losses.to(device)
        self.perturbations = self.perturbations.to(device)
        return self

    def update(self, epoch_successes: EpochSuccesses):
        loss_values_to_check = self.losses[epoch_successes.batch_indices]

        epoch_indices_to_copy_from = self.comparison_funct(
            loss_values_to_check, epoch_successes.losses
        )

        batch_indices_to_copy_to = epoch_successes.batch_indices[
            epoch_indices_to_copy_from
        ]
        self.epochs[batch_indices_to_copy_to] = epoch_successes.epoch_num
        self.losses[batch_indices_to_copy_to] = epoch_successes.losses[
            epoch_indices_to_copy_from
        ]
        self.perturbations[batch_indices_to_copy_to, :, :] = (
            epoch_successes.perturbations[epoch_indices_to_copy_from, :, :]
        )


class BatchResult:
    def __init__(
        self,
        initial_device: torch.device,
        dataset_indices: torch.tensor,
        max_seq_length: int,
        input_size: int,
    ):
        self.dataset_indices = dataset_indices
        self.first_examples = RecordedBatchExamples(
            initial_device=initial_device,
            batch_size_actual=dataset_indices.shape[0],
            max_seq_length=max_seq_length,
            input_size=input_size,
            comparison_funct=has_no_entry,
        )
        self.best_examples = RecordedBatchExamples(
            initial_device=initial_device,
            batch_size_actual=dataset_indices.shape[0],
            max_seq_length=max_seq_length,
            input_size=input_size,
            comparison_funct=is_greater_than_new_val,
        )

    def to(self, device: torch.device):
        self.dataset_indices = self.dataset_indices.to(device)
        self.first_examples = self.first_examples.to(device)
        self.best_examples = self.best_examples.to(device)

    def update(self, epoch_successes: EpochSuccesses):
        self.first_examples.update(epoch_successes=epoch_successes)
        self.best_examples.update(epoch_successes=epoch_successes)


@dataclass
class RecordedTrainerExamples:
    epochs: torch.tensor = torch.LongTensor()
    losses: torch.tensor = torch.FloatTensor()
    perturbations: torch.tensor = torch.FloatTensor()
    device: torch.device = torch.device("cpu")

    def update(self, batch_examples: RecordedBatchExamples):
        self.epochs = torch.cat((self.epochs, batch_examples.epochs), dim=0)
        self.losses = torch.cat((self.losses, batch_examples.losses), dim=0)
        self.perturbations = torch.cat(
            (self.perturbations, batch_examples.perturbations), dim=0
        )


@dataclass
class TrainerResult:
    dataset_indices: torch.tensor = torch.LongTensor()
    first_examples: RecordedTrainerExamples = RecordedTrainerExamples()
    best_examples: RecordedTrainerExamples = RecordedTrainerExamples()


def run_batch(
    dataset_start_idx: int,
    batch_size: int = 10,
    epochs_per_batch: int = 50,
    successes_per_batch: int = 3,
):
    batch_result = BatchResult(
        dataset_indices=torch.arange(
            start=dataset_start_idx, end=dataset_start_idx + batch_size
        ),
        max_seq_length=5,
        input_size=7,
    )

    for epoch_idx in range(epochs_per_batch):
        epoch_successes = EpochSuccesses(
            epoch_num=epoch_idx,
            batch_indices=torch.tensor(
                np.random.choice(
                    np.arange(batch_size),
                    size=successes_per_batch,
                    replace=False,
                )
            ),
            losses=5 * torch.abs(torch.randn(3)),
            perturbations=torch.randn(successes_per_batch, 5, 7),
        )
        batch_result.update(epoch_successes=epoch_successes)

    return batch_result


if __name__ == "__main__":
    my_trainer_result = TrainerResult()

    my_batch_size = 10
    for batch_idx in range(5):
        batch_result = run_batch(dataset_start_idx=batch_idx * my_batch_size)
        my_trainer_result.first_examples.update(
            batch_examples=batch_result.first_examples
        )
        my_trainer_result.best_examples.update(
            batch_examples=batch_result.best_examples
        )
        my_trainer_result.dataset_indices = torch.cat(
            (my_trainer_result.dataset_indices, batch_result.dataset_indices)
        )
