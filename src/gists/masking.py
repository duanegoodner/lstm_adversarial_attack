import numpy as np

arr = np.ones((3, 5, 2), dtype=bool)
print("before masking\n")
print(arr)

# Indices of rows to set to False
rows_to_set_false = [
    (0, slice(2, None)),
    (0, slice(3, None)),
    (0, slice(4, None)),
    (2, slice(3, None)),
    (2, slice(4, None)),
]

# Create a boolean mask
mask = np.zeros_like(arr, dtype=bool)
for row_idx, col_idx in rows_to_set_false:
    mask[row_idx, col_idx, :] = True

# Set the specified rows to False
arr[mask] = False
print("\nafter masking\n")
print(arr)
