import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

# Example pandas dataframe (replace this with your actual dataframe)
data = {
    'Column1': [1, 2, 3, 4],
    'Column2': [5, 6, 7, 8],
    'Column3': [9, 10, 11, 12],
    'Column4': [13, 14, 15, 16]
}

df = pd.DataFrame(data)

# Example data for subplots (replace this with your actual plotting data)
x = [1, 2, 3, 4]
y1 = [5, 3, 8, 6]
y2 = [2, 7, 1, 4]
y3 = [9, 10, 11, 12]
y4 = [13, 14, 15, 16]

# Create a 2x2 subplot grid using gridspec
gs = gridspec.GridSpec(3, 2, height_ratios=[2, 2, 1], hspace=0.5)

# Create the subplots
fig = plt.figure()
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1])
ax3 = plt.subplot(gs[1, 0])
ax4 = plt.subplot(gs[1, 1])

# Plot data on subplots
ax1.plot(x, y1, label='Plot 1')
ax2.plot(x, y2, label='Plot 2')
ax3.plot(x, y3, label='Plot 3')
ax4.plot(x, y4, label='Plot 4')

# Add legends (optional)
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()

# Convert pandas dataframe to a table
table_data = []
for col in df.columns:
    table_data.append(df[col].tolist())

# Create the table using gridspec
table_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, :], width_ratios=[1, 2])
ax_table = plt.subplot(table_gs[:, 0])
ax_table.axis('off')  # Turn off axis for the table
table = ax_table.table(cellText=table_data, colLabels=df.columns, loc='center', cellLoc='center')

# Set table properties (optional)
table.auto_set_font_size(False)
table.set_fontsize(10)

# Adjust the layout
plt.tight_layout()

# Display the plot
plt.show()
