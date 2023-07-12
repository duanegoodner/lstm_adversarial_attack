import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
data = np.random.rand(19, 10)

# Create the heatmap using Seaborn
ax = sns.heatmap(data)

# Calculate the number of ticks and labels
num_ticks = data.shape[0] + 1
tick_positions = np.linspace(0.5, num_ticks - 0.5, num=num_ticks)
tick_labels = range(0, num_ticks * 2, 2)

# Set the y-axis tick positions and labels
ax.set_yticks(tick_positions)
# ax.set_yticklabels(tick_labels)

# Display the plot
plt.show()
