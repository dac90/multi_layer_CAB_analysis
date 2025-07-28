import numpy as np
import matplotlib.pyplot as plt
from math import comb
from mpl_toolkits.mplot3d import Axes3D

# Define G function
def G(n0, n1):
    return sum(comb(n1, j) for j in range(0, n0 + 1) if j <= n1)

# Ranges for n0 and n1
n0_vals = np.arange(0, 16)  # n^{[0]} from 0 to 7
n1_vals = np.arange(0, 16)  # n^{[1]} from 0 to 7

# Create meshgrid
N0, N1 = np.meshgrid(n0_vals, n1_vals)

# Compute G values
G_vals = np.zeros_like(N0, dtype=float)
for i in range(N0.shape[0]):
    for j in range(N0.shape[1]):
        G_vals[i, j] = G(N0[i, j], N1[i, j])

# Plot 3D surface
fig = plt.figure(figsize=(16, 5))
ax = fig.add_subplot(131, projection='3d')
ax.plot_surface(N0, N1, G_vals, cmap='viridis', edgecolor='k')

ax.set_xlabel('Input Layer Dimension (n[0])')
ax.set_ylabel('Hidden Layer Neurons (n[1])')
ax.set_zlabel('Upper Bound on Non-Void Partitions')
ax.set_title('Upper Bound on Non-Void Partitions\nin a Single Hidden Layer Network')

# --- 2D plot: Fix n1 = 8 ---
n_range = np.arange(0, 17)  # allow n0 up to 16
G_n1_fixed = [G(n0, 8) for n0 in n_range]

ax2 = fig.add_subplot(132)
ax2.plot(n_range, G_n1_fixed, marker='o')
ax2.set_xlabel('Input Layer Dimension (n[0])')
ax2.set_ylabel('Upper Bound on Non-Void Partitions')
ax2.set_title('Cross-Section with 8 Neurons')
ax2.grid(True)

# --- 2D plot: Fix n0 = 8 ---
G_n0_fixed = [G(8, n1) for n1 in n_range]

ax3 = fig.add_subplot(133)
ax3.plot(n_range, G_n0_fixed, marker='o', color='orange')
ax3.set_xlabel('Hidden Layer Neurons (n[1])')
ax3.set_ylabel('Upper Bound on Non-Void Partitions')
ax3.set_title('Cross-Section with 8 Input Dimensions')
ax3.grid(True)

plt.tight_layout()
plt.show()