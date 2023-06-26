import numpy as np

arr = np.array([2, 1, 3])

# Create an empty boolean array of the desired shape
result = np.zeros((len(arr), 4, 2), dtype=bool)

# Set True values based on the number specified in the input array
rows = np.arange(result.shape[1])
comparison = rows < arr[:, np.newaxis]
comparison = comparison[..., np.newaxis]  # Reshape the array

result[np.arange(len(arr))[:, np.newaxis], rows] = comparison

print(result)
