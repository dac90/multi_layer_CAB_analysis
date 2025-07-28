import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import os
os.environ["JULIA_LOAD_PATH"] = "C:/Users/dcies/Desktop/Summer Research/CAB_analysis"
os.environ["JULIA_PROJECT"] = "C:/Users/dcies/Desktop/Summer Research/CAB_analysis"

from julia import Julia
jl = Julia(runtime="C:/Users/dcies/AppData/Local/Programs/Julia-1.11.6/bin/julia.exe", compiled_modules=False)


from julia import CAB_analysis  # Now import the module after inclusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)

# Network architecture
n = [2, 4, 4, 1]
m = 1000

# Target function
def target_function(x, y):
    return x * y - 1

# Flexible network definition
class FlexibleReLUNetwork(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Generate training data
x_vals = np.random.uniform(-5, 5, (m, 1))
y_vals = np.random.uniform(-5, 5, (m, 1))
X_np = np.hstack([x_vals, y_vals])
Y_np = target_function(x_vals, y_vals)

X = torch.tensor(X_np, dtype=torch.float64).to(device)
Y = torch.tensor(Y_np, dtype=torch.float64).to(device)

# Instantiate model
model = FlexibleReLUNetwork(n).to(device).double()

# Optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop
steps_per_epoch = 10
total_epochs = 1000

for epoch in range(total_epochs):
    for _ in range(steps_per_epoch):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    # Save params to .jlser
    CAB_analysis.save_params(model.network, epoch)
    #CAB_analysis.get_params(epoch)

    # Calculate CAB Tree (Boundary Partitions Only)
    #CAB_analysis.calculate_boundary_tree(n, 1, epoch)  # Pass hidden sizes
    #CAB_analysis.get_partition_tree(epoch)

    # Calculate CAB All (For all neurons)
    CAB_analysis.calculate_mixed_all(n, epoch)  # Pass hidden sizes
    #CAB_analysis.get_partition_all(epoch)

for epoch in range(total_epochs):
    # Plot CAB
    CAB_analysis.plot_CAB_all(epoch)  # Pass hidden sizes

CAB_analysis.create_animation()