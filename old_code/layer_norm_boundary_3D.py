import torch
import matplotlib.pyplot as plt
import numpy as np

# ===== Dummy parameters (edit these) =====
gamma = torch.tensor([1.0, 0.8, 1.2])  # LayerNorm scale
beta = torch.tensor([0.0, 0.5, -0.3])  # LayerNorm shift
W = torch.tensor([0.7, -1.0, -0.5])     # Dense layer weights
b = torch.tensor(-1.6)                  # Dense layer bias
eps = 1e-5

# ===== LayerNorm broken into steps =====
def mean_norm(x):
    mean = x.mean(dim=-1, keepdim=True)
    return x - mean

def var_norm(x):
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return x / torch.sqrt(var + eps)

def scale_and_shift(x):
    return gamma * x + beta

def dense_pre_activation(x):
    return (x * W).sum(dim=-1) + b

# ===== Pipelines for each stage =====
def pipeline_stage1(x):
    out = mean_norm(x)
    out = var_norm(out)
    out = scale_and_shift(out)
    return dense_pre_activation(out)

def pipeline_stage2(x):
    out = var_norm(x)
    out = scale_and_shift(out)
    return dense_pre_activation(out)

def pipeline_stage3(x):
    out = scale_and_shift(x)
    return dense_pre_activation(out)

def pipeline_stage4(x):
    return dense_pre_activation(x)

pipelines = [pipeline_stage1, pipeline_stage2, pipeline_stage3, pipeline_stage4]
titles = ["Before Mean Norm", "Before Var Norm", "Before Scaling", "Before Dense"]

# LaTeX axis labels for each plot
axis_labels = [
    [r"$a^{[0]}_1$", r"$a^{[0]}_2$", r"$a^{[0]}_3$"],
    [r"$a'^{[0]}_1$", r"$a'^{[0]}_2$", r"$a'^{[0]}_3$"],
    [r"$a''^{[0]}_1$", r"$a''^{[0]}_2$", r"$a''^{[0]}_3$"],
    [r"$a^{[1]}_1$", r"$a^{[1]}_2$", r"$a^{[1]}_3$"],
]

# ===== Zero-surface sampling =====
def get_zero_points(pipeline, lim=(-5, 5), res=50, tol=0.02):
    x = torch.linspace(lim[0], lim[1], res)
    y = torch.linspace(lim[0], lim[1], res)
    z = torch.linspace(lim[0], lim[1], res)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    pts = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
    vals = pipeline(pts)
    mask = torch.abs(vals) < tol
    return pts[mask]

# ===== Shape plotting helpers =====
def plot_plane(ax, normal, point, size=5, alpha=0.2, color='orange'):
    """Plot a plane from a normal vector and a point on the plane."""
    d = -point.dot(normal)
    xx, yy = np.meshgrid(np.linspace(-size, size, 20),
                         np.linspace(-size, size, 20))
    zz = (-normal[0]*xx - normal[1]*yy - d) / normal[2]
    ax.plot_surface(xx, yy, zz, alpha=alpha, color=color, rstride=1, cstride=1)

def plot_sphere(ax, center, radius, alpha=0.2, color='green'):
    """Plot a sphere."""
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha)

def plot_ellipsoid(ax, center, radii, alpha=0.2, color='red'):
    """Plot an axis-aligned ellipsoid."""
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    x = center[0] + radii[0] * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radii[1] * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radii[2] * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha)

# ===== Plotting =====
fig = plt.figure(figsize=(16, 12))
for i, (pipe, title, labels) in enumerate(zip(pipelines, titles, axis_labels), start=1):
    pts = get_zero_points(pipe)
    ax = fig.add_subplot(2, 2, i, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, alpha=0.3, c='blue')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_title(title)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])

    # Extra shapes
    if i == 2:  # Plot 2: plane through origin, normal [1,1,1]
        plot_plane(ax, np.array([1, 1, 1]) / np.sqrt(3), np.array([0, 0, 0]))
    elif i == 3:  # Plot 3: sphere at origin, radius sqrt(3)
        plot_sphere(ax, np.array([0, 0, 0]), np.sqrt(3))
    elif i == 4:  # Plot 4: ellipsoid at beta, radii gamma * sqrt(3)
        plot_ellipsoid(ax, beta.numpy(), (gamma.numpy() * np.sqrt(3)))

plt.tight_layout()
plt.show()