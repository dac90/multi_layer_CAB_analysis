import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import matplotlib.cm as cm
import matplotlib.colors as mcolors

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# Full layer size list: includes input and output layer sizes
n = [2, 4, 4, 1]  # input → hidden layers → output
L=len(n)-1
# Generate training data for f(x, y) = sin(x) * cos(y)
def target_function(x, y):
    return x*y - 1

# Obtain weights and biases

def get_weights_and_biases(model):
    W = [np.identity(n[0], dtype=float)]
    b = [np.zeros(n[0], dtype=float)]
    for layer in model.network:
        if isinstance(layer, nn.Linear):
            W.append(layer.weight.detach().cpu().numpy())
            b.append(layer.bias.detach().cpu().numpy())
    return W, b

def get_relu_activations(model, x_input):
    """
    Returns a list of 1D boolean numpy arrays, one per input, representing
    the activation state (True=active) of all ReLU units in the network.
    """
    x = x_input.clone()
    all_masks = []

    for layer in model.network:
        x = layer(x)
        if isinstance(layer, nn.ReLU):
            mask = (x > 0).detach().cpu().numpy()  # shape [batch_size, num_units]
            all_masks.append(mask)

    # Flatten ReLU outputs for each input across all layers
    flat_masks = np.hstack(all_masks)  # shape [batch_size, total_relu_units]
    return flat_masks

def CAB_direct(W, b, R):
    W_tensors = [torch.tensor(w, dtype=torch.float32, device=device) for w in W]
    b_tensors = [torch.tensor(bi, dtype=torch.float32, device=device).reshape(-1, 1) for bi in b]
    R_tensors = [torch.tensor(r, dtype=torch.float32, device=device) for r in R]

    W_tilde = torch.eye(1, device=device)
    W_tilde_list = [W_tilde.clone()]
    for i in range(1, L + 1):
        W_tilde = W_tilde @ W_tensors[L + 1 - i] @ R_tensors[L - i]
        W_tilde_list.insert(0, W_tilde.clone())

    b_tilde = torch.zeros((W_tilde.shape[0], 1), device=device)
    for j in range(1, L + 1):
        b_tilde += W_tilde_list[j] @ b_tensors[j]

    phi, _, _, _     = torch.linalg.lstsq(W_tilde, -b_tilde)
    return phi[:W_tilde.shape[1], 0].cpu().numpy()

def plot_analytical_CAB(model, epoch=None):
    device = next(model.parameters()).device

    # Grid
    x = np.linspace(-5, 5, 500)
    y = np.linspace(-5, 5, 500)
    Xg, Yg = np.meshgrid(x, y)
    points = np.vstack([Xg.ravel(), Yg.ravel()]).T
    inputs = torch.tensor(points, dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = model(inputs).cpu().numpy().reshape(500, 500)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Compute all activations once
    all_activations = get_relu_activations(model, inputs)
    unique_r_rows = np.unique(all_activations, axis=0)

    W, b = get_weights_and_biases(model)
    # Create a colormap for shading
    cmap = cm.get_cmap('tab20', len(unique_r_rows))  # or 'viridis', 'Set3', etc.

    for i, r in enumerate(unique_r_rows):
        matches = np.all(all_activations == r, axis=1)
        mask = matches.reshape(Xg.shape)

        # Plot light shading for this region
        color = cmap(i)
        ax1.contourf(Xg, Yg, mask.astype(float), levels=[0.5, 1], colors=[color], alpha=0.2)
        ax2.contourf(Xg, Yg, mask.astype(float), levels=[0.5, 1], colors=[color], alpha=0.2)

        # Continue with your CAB contour
        R = [np.identity(n[0])]
        idx = 0
        for layer_size in n[1:-1]:
            r_slice = r[idx:idx + layer_size]
            R.append(np.diag(r_slice.astype(float)))
            idx += layer_size

        phi = CAB_direct(W, b, R)
        F = phi[0] * (Xg - phi[0]) + phi[1] * (Yg - phi[1])
        F_masked = np.ma.masked_where(~mask, F)
        ax1.contour(Xg, Yg, F_masked, levels=[0], colors='black')

    ax2.contour(Xg, Yg, predictions, levels=[0], colors='red', linewidths=2)

    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_title(f'Analytical CAB at Epoch {epoch}' if epoch is not None else 'Analytical CAB')

    ax2.set_aspect('equal')
    ax2.grid(True)
    ax2.set_title(f'Empirical CAB at Epoch {epoch}' if epoch is not None else 'Empirical CAB')

    plt.show()

def plot_predictions(model, epoch=None):
    device = next(model.parameters()).device
    with torch.no_grad():
        x = np.linspace(-5, 5, 500)
        y = np.linspace(-5, 5, 500)
        Xg, Yg = np.meshgrid(x, y)
        points = np.vstack([Xg.ravel(), Yg.ravel()]).T
        inputs = torch.tensor(points, dtype=torch.float32).to(device)
        predictions = model(inputs).cpu().numpy().reshape(500, 500)

        plt.figure(figsize=(8, 6))
        plt.contourf(Xg, Yg, predictions, levels=100, cmap='viridis')
        plt.colorbar(label='Predicted Output')
        plt.contour(Xg, Yg, predictions, levels=[0], colors='red', linewidths=2)
        full_title = f'NN Approximation at Epoch {epoch}' if epoch is not None else 'NN Approximation'
        plt.title(full_title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.show()

# Define a neural network with arbitrary layer sizes
class FlexibleReLUNetwork(nn.Module):
    def __init__(self, layer_sizes):
        super(FlexibleReLUNetwork, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # No ReLU after last layer
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Instantiate the model
m = 1000
x_vals = np.random.uniform(-5, 5, (m, 1))
y_vals = np.random.uniform(-5, 5, (m, 1))
X = torch.tensor(np.hstack([x_vals, y_vals]), dtype=torch.float32)
Y = torch.tensor(target_function(x_vals, y_vals), dtype=torch.float32)
model = FlexibleReLUNetwork(layer_sizes=n)

X = X.to(device)
Y = Y.to(device)
model = model.to(device)

# Training setup
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
n_epochs = 10000
for epoch in range(n_epochs+1):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()

    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        plot_analytical_CAB(model, epoch)
        plot_predictions(model, epoch)



'''
def CAB_zeta(phi,R):
    R_phi = R @ phi
    zeta = (np.dot(phi, phi)/np.dot(R_phi, R_phi))*R_phi
    return zeta

def CAB_phi(zeta,W,b):
    bias_factor = 1-(np.dot(zeta, b)/np.dot(zeta, zeta))
    W_zeta = W.T @ zeta
    phi = bias_factor * (np.dot(zeta, zeta)/np.dot(W_zeta, W_zeta)) * W_zeta
    return phi
'''