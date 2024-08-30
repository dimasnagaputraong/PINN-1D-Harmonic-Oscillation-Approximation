import csv
import itertools
import time as tm

import matplotlib.pyplot as plt
import numpy as np
import pynvml
import torch
import torch.nn as nn
from PIL import Image
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import curve_fit
from torch.utils.data import DataLoader
from pyDOE import lhs


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize NVML
pynvml.nvmlInit()
device_index = 0  # Change this if you have multiple GPUs
handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)


class FCN_2DOF(nn.Module):
    "Defines a fully connected network for a 2DOF system with learnable k and d parameters"

    def __init__(self, N_INPUT, N_OUTPUT, hidden_layers, init_k, init_d):
        super().__init__()
        activation = nn.Tanh

        # Define the input layer
        layers = [nn.Linear(N_INPUT, hidden_layers[0]), activation()]

        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(activation())

        self.fch = nn.Sequential(*layers)

        # Define the output layer
        self.fce = nn.Linear(hidden_layers[-1], N_OUTPUT)

        # Initialize k and d parameters (for two degrees of freedom)
        self.k1 = nn.Parameter(torch.tensor(init_k[0], dtype=torch.float32))  # Initialize k1
        self.d1 = nn.Parameter(torch.tensor(init_d[0], dtype=torch.float32))  # Initialize d1
        self.k2 = nn.Parameter(torch.tensor(init_k[1], dtype=torch.float32))  # Initialize k2
        self.d2 = nn.Parameter(torch.tensor(init_d[1], dtype=torch.float32))  # Initialize d2

    def forward(self, x):
        x = self.fch(x)
        x = self.fce(x)
        return x

    def enforce_non_negative(self):
        self.k1.data.clamp_(min=0)
        self.d1.data.clamp_(min=0)
        self.k2.data.clamp_(min=0)
        self.d2.data.clamp_(min=0)


def mass_spring_damper_3dof(state, t, m1, m2, m3, d1, d2, d3, k1, k2, k3):
    x1, v1, x2, v2, x3, v3 = state  # unpack the state vector

    dx1dt = v1  # derivative of x1 is v1
    dv1dt = (-d1 * v1 - k1 * x1 + d2 * (v2 - v1) + k2 * (x2 - x1)) / m1  # acceleration for mass 1

    dx2dt = v2  # derivative of x2 is v2
    dv2dt = (d2 * (v1 - v2) + k2 * (x1 - x2) - d3 * v2 - k3 * x2 + d3 * (v3 - v2) + k3 * (x3 - x2)) / m2  # acceleration for mass 2

    dx3dt = v3  # derivative of x3 is v3
    dv3dt = (d3 * (v2 - v3) + k3 * (x2 - x3)) / m3  # acceleration for mass 3

    return [dx1dt, dv1dt, dx2dt, dv2dt, dx3dt, dv3dt]


def solve_3dof_system(m1, m2, m3, d1, d2, d3, k1, k2, k3, x0, v0, t):
    initial_state = x0 + v0
    states = odeint(mass_spring_damper_3dof, initial_state, t, args=(m1, m2, m3, d1, d2, d3, k1, k2, k3))

    # Extract positions and velocities
    x1, v1 = states[:, 0], states[:, 1]
    x2, v2 = states[:, 2], states[:, 3]
    x3, v3 = states[:, 4], states[:, 5]

    # Calculate accelerations
    a1 = (-k1 * x1 - d1 * v1 + k2 * (x2 - x1) + d2 * (v2 - v1)) / m1
    a2 = (-k2 * (x2 - x1) - d2 * (v2 - v1) + k3 * (x3 - x2) + d3 * (v3 - v2)) / m2
    a3 = (-k3 * (x3 - x2) - d3 * (v3 - v2)) / m3

    return (x1, v1, a1), (x2, v2, a2), (x3, v3, a3)


def squared_difference(input, target):
    return (input - target) ** 2


def estimate_k_d_2dof(t, y1, y2, m1, m2):
    y1 = y1.cpu().numpy().flatten()
    y2 = y2.cpu().numpy().flatten()

    def damped_oscillator_2dof(t, A1, B1, omega_n1, zeta1, A2, B2, omega_n2, zeta2):
        if zeta1 < 1:
            omega_d1 = omega_n1 * np.sqrt(1 - zeta1 ** 2)
            response1 = np.exp(-zeta1 * omega_n1 * t) * (A1 * np.cos(omega_d1 * t) + B1 * np.sin(omega_d1 * t))
        else:
            response1 = (A1 + B1 * t) * np.exp(-omega_n1 * t)

        if zeta2 < 1:
            omega_d2 = omega_n2 * np.sqrt(1 - zeta2 ** 2)
            response2 = np.exp(-zeta2 * omega_n2 * t) * (A2 * np.cos(omega_d2 * t) + B2 * np.sin(omega_d2 * t))
        else:
            response2 = (A2 + B2 * t) * np.exp(-omega_n2 * t)

        return response1 + response2

    # Initial guesses for A1, B1, omega_n1, zeta1, A2, B2, omega_n2, zeta2
    initial_guess = [y1[0], 0, 1.0, 0.5, y2[0], 0, 1.0, 0.5]

    # Fit the damped oscillator model to the data
    params, _ = curve_fit(damped_oscillator_2dof, t, np.hstack([y1, y2]), p0=initial_guess)
    A1, B1, omega_n1, zeta1, A2, B2, omega_n2, zeta2 = params

    # Calculate initial estimates for k and d
    k1 = m1 * omega_n1 ** 2
    d1 = 2 * zeta1 * np.sqrt(k1 * m1)
    k2 = m2 * omega_n2 ** 2
    d2 = 2 * zeta2 * np.sqrt(k2 * m2)

    return [k1, k2], [d1, d2]


def train_nn_2dof(x_data, y1_data, y2_data, x_physics, epochs=20000):
    pde_loss_history = []
    data_loss_history = []
    total_loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        y1h = model(x_data)
        y2h = model(x_data)
        loss1 = torch.mean((y1h - y1_data) ** 2) + torch.mean((y2h - y2_data) ** 2)

        y1hp = model(x_physics)
        y2hp = model(x_physics)
        dx1 = torch.autograd.grad(y1hp, x_physics, torch.ones_like(y1hp), create_graph=True)[0]
        dx1_2 = torch.autograd.grad(dx1, x_physics, torch.ones_like(dx1), create_graph=True)[0]
        dx2 = torch.autograd.grad(y2hp, x_physics, torch.ones_like(y2hp), create_graph=True)[0]
        dx2_2 = torch.autograd.grad(dx2, x_physics, torch.ones_like(dx2), create_graph=True)[0]

        physics = dx1_2 + (model.d1 / m1) * dx1 + (model.k1 / m1) * y1hp + \
                  dx2_2 + (model.d2 / m2) * dx2 + (model.k2 / m2) * y2hp

        loss2 = torch.mean(physics ** 2)

        loss = loss1 + (1e-4) * loss2
        loss.backward()
        optimizer.step()

        model.enforce_non_negative()

        pde_loss_history.append(loss2.item())
        data_loss_history.append(loss1.item())
        total_loss_history.append(loss.item())

        if (epoch + 1) % 500 == 0:
            print(
                f"Epoch: {epoch + 1}\tTotal Loss: {loss.item()}\tPhysics Loss: {loss2.item()}\t"
                f"Data Loss: {loss1.item()}\t"
                f"k1: {model.k1.item()}\tk2: {model.k2.item()}\t"
                f"d1: {model.d1.item()}\td2: {model.d2.item()}"
            )
            print("Current Learning Rate:", optimizer.param_groups[0]['lr'])

    # Plot the loss history
    plt.figure(figsize=(10, 6))
    plt.plot(pde_loss_history, label='PDE Loss')
    plt.plot(data_loss_history, label='Data Loss')
    plt.plot(total_loss_history, label='Overall Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.legend()
    plt.grid(True)

    with torch.no_grad():
        learned_k1 = model.k1.item()
        learned_d1 = model.d1.item()
        learned_k2 = model.k2.item()
        learned_d2 = model.d2.item()

    return [learned_k1, learned_k2], [learned_d1, learned_d2]


def calculate_damping_quality_kpi_2dof(k1, d1, m1, k2, d2, m2):
    eigenfrequency1 = np.sqrt(k1 / m1) / (2 * np.pi)
    damping_ratio1 = d1 / (2 * np.sqrt(k1 * m1))
    quality_factor1 = 1 / (2 * damping_ratio1)

    eigenfrequency2 = np.sqrt(k2 / m2) / (2 * np.pi)
    damping_ratio2 = d2 / (2 * np.sqrt(k2 * m2))
    quality_factor2 = 1 / (2 * damping_ratio2)

    return (eigenfrequency1, damping_ratio1, quality_factor1), (eigenfrequency2, damping_ratio2, quality_factor2)


# Example usage with the 3DOF system
# Define parameter ranges for 3DOF model
c1_values = np.linspace(10000, 100000, 10)
d1_values = np.linspace(500, 20000, 10)
c2_values = np.linspace(50000, 200000, 10)
d2_values = np.linspace(500, 15000, 10)
c3_values = np.linspace(200000, 500000, 10)
d3_values = np.linspace(1000, 20000, 10)

# Number of samples and parameters
num_samples = 250
num_parameters = 6

# Generate LHS samples
lhs_samples = lhs(num_parameters, samples=num_samples)

# Scale the samples to the actual parameter ranges
scaled_samples = np.zeros_like(lhs_samples)
scaled_samples[:, 0] = c1_values[0] + lhs_samples[:, 0] * (c1_values[-1] - c1_values[0])
scaled_samples[:, 1] = d1_values[0] + lhs_samples[:, 1] * (d1_values[-1] - d1_values[0])
scaled_samples[:, 2] = c2_values[0] + lhs_samples[:, 2] * (c2_values[-1] - c2_values[0])
scaled_samples[:, 3] = d2_values[0] + lhs_samples[:, 3] * (d2_values[-1] - d2_values[0])
scaled_samples[:, 4] = c3_values[0] + lhs_samples[:, 4] * (c3_values[-1] - c3_values[0])
scaled_samples[:, 5] = d3_values[0] + lhs_samples[:, 5] * (d3_values[-1] - d3_values[0])

print(scaled_samples)

start = tm.time()

# Iterate through combinations and train the 2DOF PINN
for i, combination in enumerate(scaled_samples):
    c1, d1, c2, d2, c3, d3 = combination

    print(f"Dataset {i + 1}/{len(scaled_samples)} in progress...")
    print(f"c1: {c1}\td1: {d1}\tc2: {c2}\td2: {d2}\tc3: {c3}\td3: {d3}")

    csv_file_path = 'Dataset_BMW_KPI_LHS.csv'

    def z_S(t):
        return 0

    def z_S_dot(t):
        return 0

    def quarter_car_3dof_model(t, y):
        z1, v1, z2, v2, z3, v3 = y
        dz1_dt = v1
        dv1_dt = (-c1 * z1 - d1 * v1 + c2 * (z2 - z1) + d2 * (v2 - v1)) / m1
        dz2_dt = v2
        dv2_dt = (-c2 * (z2 - z1) - d2 * (v2 - v1) + c3 * (z3 - z2) + d3 * (v3 - v2)) / m2
        dz3_dt = v3
        dv3_dt = (-c3 * (z3 - z2) - d3 * (v3 - v2)) / m3
        return [dz1_dt, dv1_dt, dz2_dt, dv2_dt, dz3_dt, dv3_dt]

    y0 = [0.1, 0, 0.1, 0, 0.1, 0]

    t_span = (0, 6)
    t = np.linspace(0, 6, 600)

    m1, m2, m3 = 537, 68, 100

    sol = solve_ivp(quarter_car_3dof_model, t_span, y0, t_eval=t)

    z1, v1 = sol.y[0], sol.y[1]
    z2, v2 = sol.y[2], sol.y[3]
    z3, v3 = sol.y[4], sol.y[5]
    t = sol.t

    a1 = np.gradient(v1, t)
    a2 = np.gradient(v2, t)
    a3 = np.gradient(v3, t)

    z_S_vals = np.array([z_S(ti) for ti in t])

    plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    plt.plot(t, z_S_vals, label='Road Input $z_S(t)$')
    plt.plot(t, z1, label='$z_{1}(t)$ - Displacement of Mass 1')
    plt.plot(t, z2, label='$z_{2}(t)$ - Displacement of Mass 2')
    plt.plot(t, z3, label='$z_{3}(t)$ - Displacement of Mass 3')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.title('Road Input and Displacement Responses')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t, v1, label='$v_{1}(t)$ - Velocity of Mass 1')
    plt.plot(t, v2, label='$v_{2}(t)$ - Velocity of Mass 2')
    plt.plot(t, v3, label='$v_{3}(t)$ - Velocity of Mass 3')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity Responses')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, a1, label='$a_{1}(t)$ - Acceleration of Mass 1')
    plt.plot(t, a2, label='$a_{2}(t)$ - Acceleration of Mass 2')
    plt.plot(t, a3, label='$a_{3}(t)$ - Acceleration of Mass 3')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.title('Acceleration Responses')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    t = np.linspace(0, 6, 600)
    x = torch.tensor(t[0:600:1], dtype=torch.float32).view(-1, 1).to(device)
    y1 = torch.tensor(z1[0:600:1], dtype=torch.float32).view(-1, 1).to(device)
    y2 = torch.tensor(z2[0:600:1], dtype=torch.float32).view(-1, 1).to(device)
    y3 = torch.tensor(z3[0:600:1], dtype=torch.float32).view(-1, 1).to(device)

    init_k, init_d = estimate_k_d_2dof(t, y1, y2, m1, m2)
    print(f"Estimated k1: {init_k[0]}, k2: {init_k[1]}")
    print(f"Estimated d1: {init_d[0]}, d2: {init_d[1]}")

    x_data = x[0:600:10]
    y1_data = y1[0:600:10]
    y2_data = y2[0:600:10]
    y3_data = y3[0:600:10]

    x_physics = torch.linspace(0, 6, 60).view(-1, 1).to(device).requires_grad_(True)

    plt.figure()
    plt.plot(x.cpu(), y1.cpu(), label="Exact solution for Mass 1")
    plt.scatter(x_data.cpu(), y1_data.cpu(), color="tab:orange", label="Training data for Mass 1")
    plt.legend()

    plt.figure()
    plt.plot(x.cpu(), y2.cpu(), label="Exact solution for Mass 2")
    plt.scatter(x_data.cpu(), y2_data.cpu(), color="tab:orange", label="Training data for Mass 2")
    plt.legend()

    plt.figure()
    plt.plot(x.cpu(), y3.cpu(), label="Exact solution for Mass 3")
    plt.scatter(x_data.cpu(), y3_data.cpu(), color="tab:orange", label="Training data for Mass 3")
    plt.legend()

    # PINN for 2DOF system
    model = FCN_2DOF(N_INPUT=1, N_OUTPUT=1, hidden_layers=[256, 128, 128, 128, 256], init_k=init_k, init_d=init_d).to(device)
    initial_lr = 1e-3
    optimizer = torch.optim.Adam([
        {'params': [param for name, param in model.named_parameters() if name not in ['k1', 'k2', 'd1', 'd2']]},
        {'params': [model.k1, model.k2, model.d1, model.d2]}], lr=initial_lr)

    learned_k, learned_d = train_nn_2dof(x_data, y1_data, y2_data, x_physics)

    # Calculate the damping quality KPI for the 2DOF system
    kpi1, kpi2 = calculate_damping_quality_kpi_2dof(learned_k[0], learned_d[0], m1, learned_k[1], learned_d[1], m2)

    print(
        f"Eigenfrequency1: {kpi1[0]}\tDamping ratio1: {kpi1[1]}\tQuality factor1: {kpi1[2]}\t"
        f"Eigenfrequency2: {kpi2[0]}\tDamping ratio2: {kpi2[1]}\tQuality factor2: {kpi2[2]}"
    )

    # Save the results to the CSV file
    results = [c1, d1, c2, d2, c3, d3, learned_k[0], learned_k[1], learned_d[0], learned_d[1], kpi1[0], kpi1[1], kpi2[0], kpi2[1]]
    save_results_to_csv(csv_file_path, results)

end = tm.time()
training_time = end - start
print(training_time)
