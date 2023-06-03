import torch
from dataclasses import dataclass


@dataclass
class BatchResult:
    dataset_indices: torch.tensor
    first_loss: torch.tensor
    best_loss: torch.tensor


@dataclass
class EpochResult:
    batch_indices: torch.tensor
    loss_vals: torch.tensor


def record_first_losses(batch_res: BatchResult, epoch_res: EpochResult):
    values_to_check = batch_res.first_loss[epoch_res.batch_indices]
    epoch_indices_to_update = torch.where(values_to_check == float("inf"))
    batch_indices_to_update = epoch_res.batch_indices[epoch_indices_to_update]
    batch_res.first_loss[batch_indices_to_update] = epoch_res.loss_vals[
        epoch_indices_to_update
    ]


my_batch_res = BatchResult(
    dataset_indices=torch.tensor([4, 5, 6, 7], dtype=torch.long),
    first_loss=torch.tensor([float("inf"), 0.7, float("inf"), 0.3]),
    best_loss=torch.tensor([float("inf"), 0.5, float("inf"), 0.3]),
)


my_epoch_res = EpochResult(
    batch_indices=torch.tensor([0, 1, 3], dtype=torch.long),
    loss_vals=torch.tensor([0.5, 0.4, 0.5]),
)

record_first_losses(
    batch_res=my_batch_res, epoch_res=my_epoch_res
)
