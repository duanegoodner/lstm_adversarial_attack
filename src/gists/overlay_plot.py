import torch
import matplotlib.pyplot as plt

# create a 2-d tensor of floats
data = torch.randn(10, 24)

# create a time vector for the x-axis
times = torch.arange(0, 24)

# create a list of measurement names
measurements = ['measurement_1', 'measurement_2', 'measurement_3', 'measurement_4',
                'measurement_5', 'measurement_6', 'measurement_7', 'measurement_8',
                'measurement_9', 'measurement_10']

# create a new figure
fig, ax = plt.subplots()

# plot each row of the data as a separate line with a label
for i in range(data.shape[0]):
    ax.plot(times, data[i], label=measurements[i])

# set the x-axis label
ax.set_xlabel('Time (hours)')

# set the y-axis label
ax.set_ylabel('Data')

# set the title
ax.set_title('Data vs Time')

# create the legend
ax.legend()

# show the plot
plt.show()
