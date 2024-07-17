import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the CSV file, skipping the first row (header)
csv_file_path = 'Dataset_BMW_KPI.csv'
data = pd.read_csv(csv_file_path, header=None, skiprows=1)

# Assuming the last three columns are the KPIs and the first four columns are the parameters
eigenfrequency = pd.to_numeric(data.iloc[:, -3], errors='coerce')
damping_ratio = pd.to_numeric(data.iloc[:, -2], errors='coerce')
quality_factor = pd.to_numeric(data.iloc[:, -1], errors='coerce')

# Prepare the data for plotting
kpi_values = np.array([eigenfrequency, damping_ratio, quality_factor]).T

# Find the reference point (last row)
reference_point = kpi_values[-1]

# Find the optimum point (eigenfrequency between 1 and 2 Hz and damping ratio nearest to 0.5)
mask = (eigenfrequency >= 1) & (eigenfrequency <= 2)
filtered_kpis = kpi_values[mask]
if filtered_kpis.size > 0:
    closest_idx = np.argmin(np.abs(filtered_kpis[:, 1] - 0.5))
    optimum_point = filtered_kpis[closest_idx]
else:
    optimum_point = np.array([None, None, None])

# Find the corresponding first four columns for the optimum point
optimum_index = np.where((kpi_values == optimum_point).all(axis=1))[0]
if optimum_index.size > 0:
    optimum_params = data.iloc[optimum_index[0], :4].values
else:
    optimum_params = [None, None, None, None]

# Create the 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of all KPI values
sc = ax.scatter(kpi_values[:, 0], kpi_values[:, 1], kpi_values[:, 2], c='b', cmap='viridis', alpha=0.6)

# Plot the reference point with a red dot
ax.scatter(reference_point[0], reference_point[1], reference_point[2], c='r', s=150, edgecolor='black', marker='o', label='BMW Point')

# Plot the optimum point with a green dot if found
if None not in optimum_point:
    ax.scatter(optimum_point[0], optimum_point[1], optimum_point[2], c='g', s=150, edgecolor='black', marker='o', label='Optimum Point')

# Labels and title
ax.set_xlabel('Eigenfrequency (Hz)')
ax.set_ylabel('Damping Ratio')
ax.set_zlabel('Quality Factor')
plt.title('3D Scatter Plot of KPIs')
plt.legend()
plt.colorbar(sc, label='KPI Value')
plt.show()

# Print the corresponding first four columns of the optimum point
print(f"Optimum parameters: {optimum_params}")
print(f"Optimum KPIs: Eigenfrequency = {optimum_point[0]}, Damping Ratio = {optimum_point[1]}, Quality Factor = {optimum_point[2]}")







# Split the dataset into inputs and outputs
X = data[['c_F', 'd_F', 'c_R', 'd_R']].values
y = data[['Eigenfrequency', 'DampingRatio', 'QualityFactor']].values

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Create a dataset and data loaders
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Define the Neural Network Architecture
class KPIModel(nn.Module):
    def __init__(self):
        super(KPIModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.network(x)


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model, loss function, and optimizer
model = KPIModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with loss history tracking
EPOCHS = 10000
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Evaluate the model on the entire dataset
model.eval()
with torch.no_grad():
    predictions = model(X_tensor.to(device)).cpu().numpy()

# Calculate the percentage loss for each input data point
percentage_loss = 100 * np.abs((predictions - y) / y)

# Plot the percentage loss for each input data point
plt.figure(figsize=(12, 6))
for i in range(3):  # There are 3 KPI columns
    plt.plot(percentage_loss[:, i], label=f'KPI {i + 1}')

plt.xlabel('Sample Index')
plt.ylabel('Percentage Loss')
plt.title('Percentage Loss of KPIs for Each Input Data Point')
plt.legend(['Eigenfrequency', 'Damping Ratio', 'Quality Factor'])
plt.grid(True)
plt.show()

# Plot the loss history
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss History')
plt.legend()
plt.grid(True)
plt.show()
