import numpy as np


def sub_arrays_argmax(full_array: np.array, axis: int) -> np.array:
    idx = full_array.reshape(full_array.shape[axis], -1).argmax(axis=-1)
    return np.unravel_index(idx, full_array.shape[-2:])

x = np.random.randint(10, size=(4,3,3))
print(x)
result = sub_arrays_argmax(full_array=x, axis=0)
print(result)

