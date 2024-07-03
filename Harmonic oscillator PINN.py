from PIL import Image
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import pynvml

# Initialize NVML
pynvml.nvmlInit()
device_index = 0  # Change this if you have multiple GPUs
handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)


def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000 / fps), loop=loop)

def plot_result(x, y, x_data, y_data, yh, xp=None):
    "Pretty plot training results"
    plt.figure(figsize=(8, 4))
    plt.plot(x.cpu(), y.cpu(), color="grey", linewidth=2, alpha=0.8, label="Exact solution")
    plt.plot(x.cpu(), yh.cpu(), color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
    plt.scatter(x_data.cpu(), y_data.cpu(), s=60, color="tab:orange", alpha=0.4, label='Training data')
    if xp is not None:
        plt.scatter(xp.cpu(), -0 * torch.ones_like(xp.cpu()), s=60, color="tab:green", alpha=0.4, label='Physics loss training locations')
    l = plt.legend(loc=(1.01, 0.34), frameon=False, fontsize="large")
    plt.setp(l.get_texts(), color="k")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1.1, 1.1)
    plt.text(1.065, 0.7, "Training step: %i" % (i + 1), fontsize="xx-large", color="k")
    plt.axis("off")

def oscillator(m, k, d, x):
    """Defines the analytical solution to the 1D underdamped harmonic oscillator problem."""
    w0 = np.sqrt(k / m)
    assert d < 2 * np.sqrt(k * m)  # Ensure it's underdamped
    w = np.sqrt(w0 ** 2 - (d / (2 * m)) ** 2)
    phi = np.arctan(-d / (2 * m * w))
    A = 1 / (2 * np.cos(phi))
    cos = torch.cos(phi + w * x)
    exp = torch.exp(-d / (2 * m) * x)
    y = exp * 2 * A * cos
    return y

# Define the road input (hole of -0.1 meter)
def z_S(t):
    if t < 1:
        return -0.1
    else:
        return -0.1

# Derivative of road input
def z_S_dot(t):
    return 0  # Assuming a step input for simplicity

# Define the ODE system
def quarter_car_model(t, y):
    z_F, v_F, z_R, v_R = y
    dz_F_dt = v_F
    dz_R_dt = v_R
    dv_F_dt = (-c_F * (z_F - z_R) - d_F * (v_F - v_R)) / m_F
    dv_R_dt = (c_F * (z_F - z_R) + d_F * (v_F - v_R) - c_R * (z_R - z_S(t)) - d_R * (v_R - z_S_dot(t))) / m_R
    return [dz_F_dt, dv_F_dt, dz_R_dt, dv_R_dt]

class FCN(nn.Module):
    "Defines a fully connected network with learnable k and d parameters"
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()
        )
        self.fch = nn.Sequential(
            *[nn.Sequential(nn.Linear(N_HIDDEN, N_HIDDEN), activation()) for _ in range(N_LAYERS - 1)]
        )
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        self.k = nn.Parameter(torch.tensor(20.0))  # Initialize k
        self.d = nn.Parameter(torch.tensor(1.0))  # Initialize d

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Parameters
m = 0.1  # kg, mass of the body

# Get the analytical solution over the full domain
x = torch.linspace(0, 1, 500).view(-1, 1).to(device)
y = oscillator(m, 20, 0.1, x).view(-1, 1).to(device)  # Use the initial k and d for the exact solution
print(x.shape, y.shape)

# Sample the whole curve
x_data = x[0:500:10]
y_data = y[0:500:10]
v_data = v[0:500:10]
print(x_data.shape, y_data.shape)


plt.figure()
plt.plot(x.cpu(), y.cpu(), label="Exact solution")
plt.scatter(x_data.cpu(), y_data.cpu(), color="tab:orange", label="Training data")
plt.legend()
plt.show()

x_physics = torch.linspace(0, 5, 50).view(-1, 1).to(device).requires_grad_(True)  # Sample locations over the problem domain

torch.manual_seed(123)
model = FCN(1, 1, 32, 3).to(device)
optimizer = torch.optim.Adam([
    {'params': [param for name, param in model.named_parameters() if name not in ['k', 'd']]},
    {'params': [model.k, model.d]}
], lr=1e-3)
EPOCH = 20000
gpu_utilizations = []

start = time.time()

files = []
for i in range(EPOCH):

    optimizer.zero_grad()

    # Compute the "data loss"
    yh = model(x_data)
    loss1 = torch.mean((yh - y_data) ** 2)  # Use mean squared error

    # Compute the "physics loss"
    yhp = model(x_physics)
    dx = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0]  # Computes dy/dx
    dx2 = torch.autograd.grad(dx, x_physics, torch.ones_like(dx), create_graph=True)[0]  # Computes d^2y/dx^2
    physics = dx2 + (model.d / m) * dx + (
            model.k / m) * yhp  # Computes the residual of the 1D harmonic oscillator differential equation
    loss2 = (1e-4) * torch.mean(physics ** 2)

    # Backpropagate joint loss
    loss = loss1 + loss2  # Add two loss terms together
    loss.backward()
    optimizer.step()

    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    gpu_utilizations.append(utilization)

    # Plot the result as training progresses
    if (i + 1) % 500 == 0:
        print(
            f"Epoch: {i + 1}\tTotal Loss: {loss.item()}\tPhysics Loss: {loss2.item()}\t"
            f"Data Loss: {loss1.item()}\t"
            f"k: {model.k.item()}\td: {model.d.item()}"
        )

        yh = model(x).detach()
        xp = x_physics.detach()

        plot_result(x, y, x_data, y_data, yh, xp)

        file = "plots_pinn1d/pinn_%.8i.png" % (i + 1)
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        files.append(file)

        if (i + 1) % 2000 == 0:
            plt.show()
        else:
            plt.close("all")


save_gif_PIL("plots_pinn1d/pinn.gif", files, fps=20, loop=0)


end = time.time()
training_time = end - start
print(training_time)

plt.plot(gpu_utilizations)
plt.xlabel('Epoch')
plt.ylabel('GPU Utilization (%)')
plt.title('GPU Utilization over Time')
plt.show()


# Plot the solution of the harmonic oscillator using learned k and d
with torch.no_grad():
    learned_k = model.k.item()
    learned_d = model.d.item()
    y_learned = oscillator(m, learned_k, learned_d, x).view(-1, 1)

plt.figure(figsize=(8, 4))
plt.plot(x.cpu(), y.cpu(), color="grey", linewidth=2, alpha=0.8, label="Exact solution")
plt.plot(x.cpu(), y_learned.cpu(), color="tab:red", linewidth=2, alpha=0.8, label="Learned solution")
plt.scatter(x_data.cpu(), y_data.cpu(), s=60, color="tab:orange", alpha=0.4, label='Training data')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.title('Comparison of Exact and Learned Solutions')
plt.legend()
plt.grid(True)
plt.show()
