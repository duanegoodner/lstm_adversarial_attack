import numpy as np

# Assuming you have a sparse 3D NumPy array named 'sparse_array'
# Let's say it has shape (sub_arrays, rows, columns)
# Create a sample sparse array
sparse_array = np.array([[[0, 0, 0], [1, 2, 0], [0, 0, 0]],
                         [[0, 3, 0], [0, 0, 0], [4, 0, 5]],
                         [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=np.float64)

# Find the minimum nonzero value within each sub-array along axis 0
min_nonzero_values = np.min(np.where(sparse_array != 0, sparse_array, np.inf), axis=(1, 2))

print(min_nonzero_values)

