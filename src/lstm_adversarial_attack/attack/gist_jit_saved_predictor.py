import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(5, 1)

    def forward(self, x):
        return self.fc(x)

# Create an instance of your model
model = SimpleModel()

# Provide an example input tensor for tracing
# example_input = torch.tensor([[1.0]])
example_input = torch.randn(16, 5)

# Trace the model
traced_model = torch.jit.trace(model, example_input)

# Save the traced script module
torch.jit.save(traced_model, 'traced_model.pt')

# Define your regular PyTorch model for module A
class ModuleA(nn.Module):
    def __init__(self):
        super(ModuleA, self).__init__()
        self.fc = nn.Linear(in_features=5, out_features=5)  # 5 input and output features

    def forward(self, x):
        return self.fc(x)

# Load the traced TorchScript model for module B
module_b = torch.jit.load('traced_model.pt')

# Combine both modules in a Sequential
sequential_model = nn.Sequential(ModuleA(), module_b)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(sequential_model.parameters(), lr=0.01)

# Create a simple dataset
class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Example data and targets
data = torch.randn(100, 5)  # 100 samples with 5 features each
targets = torch.randn(100, 5)  # Corresponding targets

# Create a DataLoader for training
batch_size = 16
train_dataset = CustomDataset(data, targets)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, targets in train_dataloader:
        optimizer.zero_grad()
        outputs = sequential_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')