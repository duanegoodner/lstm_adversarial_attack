# Based on: https://stackoverflow.com/a/57833770

import torch

B = 2


def get_2d_tensor_of_1d_slices_from(
    tensor_3d: torch.tensor, indices: torch.tensor
):
    return tensor_3d[torch.arange(tensor_3d.shape[0]), indices, :].squeeze()


M = torch.tensor(
    [
        [
            [0.0612, 0.7385],
            [0.7675, 0.3444],
            [0.9129, 0.7601],
            [0.0567, 0.5602],
        ],
        [
            [0.5450, 0.3749],
            [0.4212, 0.9243],
            [0.1965, 0.9654],
            [0.7230, 0.6295],
        ],
    ]
)

my_indices = torch.tensor([3, 0])
tensor_of_rows = get_2d_tensor_of_1d_slices_from(
    tensor_3d=M, indices=my_indices
)
print(tensor_of_rows)
