import torch
from dataclasses import dataclass


@dataclass
class EpochSuccesses:
    epoch_num: int
    batch_indices: torch.tensor
    success_loss_vals: torch.tensor
    success_perturbations: torch.tensor


class BatchResult:
    def __init__(
        self,
        dataset_indices: torch.tensor,
        max_seq_length: int,
        input_size: int,
    ):
        batch_size_actual = dataset_indices.shape[0]

        self.dataset_indices = dataset_indices

        self.epoch_first_ex = torch.empty(
            batch_size_actual, dtype=torch.long
        ).fill_(-1)
        self.epoch_best_ex = torch.clone(self.epoch_first_ex)

        self.loss_first_ex = torch.empty(batch_size_actual).fill_(float("inf"))
        self.loss_best_ex = torch.clone(self.loss_first_ex)

        self.perturbation_first_ex = torch.zeros(
            size=(batch_size_actual, max_seq_length, input_size)
        )
        self.perturbation_best_ex = torch.clone(self.perturbation_first_ex)

    def update_first_examples(self, epoch_successes: EpochSuccesses):
        loss_values_to_check = self.loss_first_ex[
            epoch_successes.batch_indices
        ]

        # torch.where returns a tuple & we want 1st element of that tuple
        epoch_indices_to_copy_from = torch.where(
            loss_values_to_check == float("inf")
        )[0]

        batch_indices_to_copy_to = epoch_successes.batch_indices[
            epoch_indices_to_copy_from
        ]
        self.loss_first_ex[batch_indices_to_copy_to] = (
            epoch_successes.success_loss_vals[epoch_indices_to_copy_from]
        )
        self.epoch_first_ex[batch_indices_to_copy_to] = (
            epoch_successes.epoch_num
        )
        self.perturbation_first_ex[batch_indices_to_copy_to, :, :] = (
            epoch_successes.success_perturbations[
                epoch_indices_to_copy_from, :, :
            ]
        )


if __name__ == "__main__":
    my_dataset_indices = torch.arange(start=10, end=20)

    my_batch_result = BatchResult(
        dataset_indices=my_dataset_indices, max_seq_length=5, input_size=7
    )

    my_epoch_successes = EpochSuccesses(
        epoch_num=0,
        batch_indices=torch.tensor([2, 5, 9], dtype=torch.long),
        success_loss_vals=torch.tensor([10.4, 8.5, 9.2]),
        success_perturbations=torch.randn((3, 5, 7)),
    )

    my_batch_result.update_first_examples(epoch_successes=my_epoch_successes)
