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

def squared_difference(input, target):
    return (input - target) ** 2

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
    plt.xlim(-0.05, 6.05)
    plt.ylim(-0.5, 0.5)
    plt.text(1.065, 0.7, "Training step: %i" % (i + 1), fontsize="xx-large", color="k")
    plt.axis("off")

def estimate_k_d(t, y, m):
    # Ensure y is a numpy array for easier manipulation
    y = y.cpu().numpy().flatten()

    # Estimate the period (T)
    peaks_indices = (np.diff(np.sign(np.diff(y))) < 0).nonzero()[0] + 1  # indices of peaks
    if len(peaks_indices) < 2:
        raise ValueError("Not enough peaks found in the dataset to estimate the period.")

    T = np.mean(np.diff(t[peaks_indices]))  # Average time between peaks

    # Estimate the natural frequency (omega_n)
    omega_n = 2 * np.pi / T

    # Estimate the damping ratio (zeta) using logarithmic decrement
    peak_amplitudes = y[peaks_indices]
    delta = np.mean(np.log(peak_amplitudes[:-1] / peak_amplitudes[1:]))
    zeta = delta / np.sqrt(4 * np.pi ** 2 + delta ** 2)

    # Calculate initial estimates for k and d
    k = m * omega_n ** 2
    d = 2 * zeta * np.sqrt(k * m)

    return k, d

def oscillator(m, k, d, x):
    """Defines the analytical solution to the 1D underdamped harmonic oscillator problem with zero initial displacement."""
    w0 = np.sqrt(k / m)
    assert d < 2 * np.sqrt(k * m)  # Ensure it's underdamped
    w = np.sqrt(w0 ** 2 - (d / (2 * m)) ** 2)
    phi = np.arctan(-d / (2 * m * w))

    # Adjust amplitude A to ensure initial displacement y(0) = 0
    A = 1 / (2 * np.cos(phi))

    cos = torch.cos(w * x)
    exp = torch.exp(-d / (2 * m) * x)
    y = exp * 2 * A * cos - 2 * A
    return y

# Define the road input (hole of -0.1 meter)
def z_S(t):
    return 0

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
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, init_k, init_d):
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
        self.k = nn.Parameter(torch.tensor(init_k))  # Initialize k
        self.d = nn.Parameter(torch.tensor(init_d))  # Initialize d

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

    def enforce_non_negative(self):
        self.k.data.clamp_(min=0)
        self.d.data.clamp_(min=0)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Parameters
m_F = 537  # kg, mass of the body
m_R = 68    # kg, mass of the tire
c_F = 41332  # N/m, stiffness of the suspension
d_F = 1224   # Ns/m, damping of the suspension
c_R = 365834  # N/m, stiffness of the tire
d_R = 80    # Ns/m, damping of the tire

# Initial conditions: [z_BO, v_BO, z_T, v_T]
y0 = [0.1, 0, 0.1, 0]

# Time span
t_span = (0, 6)  # simulate for 6 seconds
t = np.linspace(0, 6, 600)

# Solve the ODE
sol = solve_ivp(quarter_car_model, t_span, y0, t_eval=t)

# Extract the solution
z_F = sol.y[0]
v_F = sol.y[1]
z_R = sol.y[2]
v_R = sol.y[3]
t = sol.t

# Compute accelerations
a_F = np.gradient(v_F, t)
a_R = np.gradient(v_R, t)

# Compute the step input
z_S_vals = np.array([z_S(ti) for ti in t])

# Plotting
plt.figure(figsize=(12, 10))

# Plot road input and displacements in the same figure
plt.subplot(3, 1, 1)
plt.plot(t, z_S_vals, label='Road Input $z_S(t)$')
plt.plot(t, z_F, label='$z_{F}(t)$ - Displacement of Car Body')
plt.plot(t, z_R, label='$z_R(t)$ - Displacement of Tire')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.title('Road Input and Displacement Responses')
plt.legend()
plt.grid(True)

# Plot velocities
plt.subplot(3, 1, 2)
plt.plot(t, v_F, label='$v_{F}(t)$ - Velocity of Car Body')
plt.plot(t, v_R, label='$v_R(t)$ - Velocity of Tire')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity Responses')
plt.legend()
plt.grid(True)

# Plot accelerations
plt.subplot(3, 1, 3)
plt.plot(t, a_F, label='$a_{F}(t)$ - Acceleration of Car Body')
plt.plot(t, a_R, label='$a_R(t)$ - Acceleration of Tire')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.title('Acceleration Responses')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

t = np.linspace(0, 6, 600)
x = torch.tensor(t[0:600:1], dtype=torch.float32).view(-1,1).to(device)
y = torch.tensor(z_F[0:600:1], dtype=torch.float32).view(-1,1).to(device)
v = torch.tensor(v_F[0:600:1], dtype=torch.float32).view(-1,1).to(device)

# Parameters
m = 1  # kg, mass of the body

# Estimate initial k and d
init_k, init_d = estimate_k_d(t, y, m)
print(f"Estimated k: {init_k}")
print(f"Estimated d: {init_d}")

# Get the analytical solution over the full domain
#x = torch.linspace(0, 1, 500).view(-1, 1).to(device)
#y = oscillator(m, 20, 0.1, x).view(-1, 1).to(device)  # Use the initial k and d for the exact solution
print(x.shape, y.shape)

# Sample the whole curve
x_data = x[0:600:10]
y_data = y[0:600:10]
v_data = v[0:600:10]
print(x_data.shape, y_data.shape)

# create DataLoader, then take one batch
#loader = DataLoader(list(zip(x_data, y_data, v_data)),
#                    shuffle=True, batch_size=5000, pin_memory=False)

plt.figure()
plt.plot(x.cpu(), y.cpu(), label="Exact solution")
plt.scatter(x_data.cpu(), y_data.cpu(), color="tab:orange", label="Training data")
plt.legend()
plt.show()

x_physics = torch.linspace(0, 6, 60).view(-1, 1).to(device).requires_grad_(True)  # Sample locations over the problem domain

torch.manual_seed(123)
model = FCN(1, 1, 50, 3, init_k, init_d).to(device)
initial_lr = 1e-3
#target_lr = 1e-4
#update_interval = 5000
#gamma = (target_lr / initial_lr) ** (1 / (30000 / update_interval))
optimizer = torch.optim.Adam([
    {'params': [param for name, param in model.named_parameters() if name not in ['k', 'd']]},
    {'params': [model.k, model.d]}], lr=initial_lr)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=update_interval, gamma=gamma)
EPOCH = 30000


pde_loss_history = []
velocity_loss_history = []
data_loss_history = []
total_loss_history = []

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
    physics = dx2 + (model.d / m) * dx + (model.k / m) * yhp  # Computes the residual of the 1D harmonic oscillator differential equation
    loss2 = torch.mean(physics ** 2)

    #loss3 = torch.mean((vh - v_data) ** 2)

    # Backpropagate joint loss
    loss = loss1 + (1e-4) * loss2  # Add two loss terms together
    loss.backward()
    optimizer.step()
    #scheduler.step()

    # Enforce non-negative constraints on k and d
    model.enforce_non_negative()

    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    gpu_utilizations.append(utilization)

    # Store the loss values
    pde_loss_history.append(loss2.item())
    data_loss_history.append(loss1.item())
    #velocity_loss_history.append(loss3.item())
    total_loss_history.append(loss.item())

    # Plot the result as training progresses
    if (i + 1) % 500 == 0:
        print(
            f"Epoch: {i + 1}\tTotal Loss: {loss.item()}\tPhysics Loss: {loss2.item()}\t"
            f"Data Loss: {loss1.item()}\t"
            f"k: {model.k.item()}\td: {model.d.item()}"
        )
        print("Current Learning Rate:", optimizer.param_groups[0]['lr'])

        yh = model(x).detach()
        xp = x_physics.detach()

        plot_result(x, y, x_data, y_data, yh, xp)

        file = "plots_pinn1d_Viertelfahrzeugmodel/pinn_%.8i.png" % (i + 1)
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        files.append(file)

        if (i + 1) % 2000 == 0:
            #plt.show()
            plt.close("all")
        else:
            #plt.close("all")
            plt.close("all")


save_gif_PIL("plots_pinn1d_Viertelfahrzeugmodel/pinn_Viertelfahrzeugmodel.gif", files, fps=20, loop=0)


end = time.time()
training_time = end - start
print(training_time)

plt.plot(gpu_utilizations)
plt.xlabel('Epoch')
plt.ylabel('GPU Utilization (%)')
plt.title('GPU Utilization over Time')
plt.show()

# Plot the loss history
plt.figure(figsize=(10, 6))
plt.plot(pde_loss_history, label='PDE Loss')
plt.plot(data_loss_history, label='Data Loss')
plt.plot(velocity_loss_history, label='Velocity Loss')
plt.plot(total_loss_history, label='Overall Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss History')
plt.legend()
plt.grid(True)
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
file = "plots_pinn1d_Viertelfahrzeugmodel/pinn_parameter_result.png"
plt.savefig(file)
plt.show()
