import time

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.linalg import expm
from torch.utils.data import DataLoader
import pynvml

# plt.style.use('dark_background')

# Initialize NVML
pynvml.nvmlInit()
device_index = 0  # Change this if you have multiple GPUs
handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)


# Function to safely convert a tensor to numpy array
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().numpy() if x.requires_grad else x.numpy()
    return x


def derivative(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]

def solve_spring_mass_damper_eigenvalue(m1, m2, k1, k2, k3, d1, d2, d3, t):
    # Constructing the coefficient matrix
    A = np.array([
        [0, 1, 0, 0],
        [-(k1 + k2) / m1, -(d1 + d2) / m1, k2 / m1, d2 / m1],
        [0, 0, 0, 1],
        [k2 / m2, d2 / m2, -(k2 + k3) / m2, -(d2 + d3) / m2]
    ])

    # Eigenvalue decomposition of A
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Construct diagonal matrix of eigenvalues
    Lambda = np.diag(eigenvalues)

    # Compute the matrix exponential of Lambda*t
    exp_Lambda_t = np.array([expm(Lambda * ti) for ti in t])

    # Compute the solution using eigenvalue decomposition
    V = eigenvectors
    V_inv = np.linalg.inv(V)
    x0 = np.array([0, 10, 0, -10])  # initial_u1, initial_v1, initial_u2, initial_v2
    x_t = np.array([np.dot(np.dot(V, exp_Lambda_t[i]), np.dot(V_inv, x0)) for i in range(len(t))])

    # Extract displacement, velocity, and acceleration for u1 and u2
    u1_t = x_t[:, 0]
    u2_t = x_t[:, 2]
    v1_t = x_t[:, 1]
    v2_t = x_t[:, 3]
    a1_t = np.gradient(v1_t, t)
    a2_t = np.gradient(v2_t, t)

    # Plot displacement, velocity, and acceleration over time in subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))

    axs[0].plot(t, u1_t, label='u1(t)')
    axs[0].plot(t, u2_t, label='u2(t)')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Displacement')
    axs[0].set_title('Displacement over Time')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(t, v1_t, label='v1(t)')
    axs[1].plot(t, v2_t, label='v2(t)')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Velocity')
    axs[1].set_title('Velocity over Time')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(t, a1_t, label='a1(t)')
    axs[2].plot(t, a2_t, label='a2(t)')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Acceleration')
    axs[2].set_title('Acceleration over Time')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    return u1_t, u2_t, v1_t, v2_t


# Parameters for generating the ground truth data
m1, m2 = 1.0, 1.0
k1, k2, k3 = 10.0, 50.0, 25.0
d1, d2, d3 = 2.0, 1.0, 3.0
t = np.linspace(0, 3, 300)

u1_t, u2_t, v1_t, v2_t = solve_spring_mass_damper_eigenvalue(m1, m2, k1, k2, k3, d1, d2, d3, t)

# Sampled ground truth data
x_ground_truth = torch.tensor(t[0:300:2], dtype=torch.float32, device=torch.device("cpu"))
y1_ground_truth = torch.tensor(u1_t[0:300:2], dtype=torch.float32, device=torch.device("cpu"))
y2_ground_truth = torch.tensor(u2_t[0:300:2], dtype=torch.float32, device=torch.device("cpu"))
v1_ground_truth = torch.tensor(v1_t[0:300:2], dtype=torch.float32, device=torch.device("cpu"))
v2_ground_truth = torch.tensor(v2_t[0:300:2], dtype=torch.float32, device=torch.device("cpu"))

# create DataLoader, then take one batch
loader = DataLoader(list(zip(x_ground_truth, y1_ground_truth, y2_ground_truth, v1_ground_truth, v2_ground_truth)),
                    shuffle=True, batch_size=5000, pin_memory=False)


# plt.plot(t, u1_t, label='u1(t)')
# plt.plot(t, u2_t, label='u2(t)')
# plt.plot(x_ground_truth, y1_ground_truth, 'o')
# plt.plot(x_ground_truth, y2_ground_truth, 'o')
# plt.legend(['Ground Truth u1', 'Ground Truth u2', 'Sampled Data u1', 'Sampled Data u2'])
# plt.show()

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu1 = nn.Parameter(torch.tensor([1.0]))
        self.mu2 = nn.Parameter(torch.tensor([1.0]))
        self.mu3 = nn.Parameter(torch.tensor([1.0]))
        self.omega1 = nn.Parameter(torch.tensor([1.0]))
        self.omega2 = nn.Parameter(torch.tensor([1.0]))
        self.omega3 = nn.Parameter(torch.tensor([1.0]))
        self.model = nn.Sequential(
            nn.Linear(1, 128),
            nn.Sigmoid(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
            # nn.Linear(128, 128),
            # nn.Sigmoid(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.model(x)

    def enforce_non_negative(self):
        # Clamping parameters to ensure non-negativity
        self.mu1.data.clamp_(min=0)
        self.mu2.data.clamp_(min=0)
        self.mu3.data.clamp_(min=0)
        self.omega1.data.clamp_(min=0)
        self.omega2.data.clamp_(min=0)
        self.omega3.data.clamp_(min=0)


EPOCHS = 5000
gpu_utilizations = []

# Define the initial and target learning rates
initial_lr = 1e-2
target_lr = 1e-3

# Define the number of epochs between each learning rate update
update_interval = 5000

# Calculate the gamma value for the scheduler
gamma = (target_lr / initial_lr) ** (1 / (20000 / update_interval))

# DEVICE = "cpu"
DEVICE = torch.device("cuda")
print(DEVICE)

# model = torch.load('PINN_EDAG_DN.pkl')
model = PINN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=update_interval, gamma=gamma)
MSE_LOSS = nn.MSELoss()

# Initial loss weights
Weight_PDE = nn.Parameter(torch.tensor([1.0], device=DEVICE, requires_grad=True))
Weight_MSE = nn.Parameter(torch.tensor([1.0], device=DEVICE, requires_grad=True))
Weight_IC = nn.Parameter(torch.tensor([1.0], device=DEVICE, requires_grad=True))

# Optimizer for the loss weights
optimizer_weights = torch.optim.Adam([Weight_PDE, Weight_MSE, Weight_IC], lr=initial_lr)

PDE_POINTS = 500


def PDE_loss(n):

    mu = [getattr(model, f'mu{i}') for i in range(1, n + 2)]
    omega = [getattr(model, f'omega{i}') for i in range(1, n + 2)]
    m = [1.0, 1.0]

    t = torch.linspace(0, 3, PDE_POINTS, requires_grad=True, device=torch.device(DEVICE)).view(-1, 1)
    x = model(t)
    x = [x[:, i].view(-1, 1) for i in range(n)]

    # Calculate the derivatives
    x_t = [derivative(x_i, t) for x_i in x]
    x_tt = [derivative(x_i_t, t) for x_i_t in x_t]


    ODE_output = []

    for i in range(n):
        if i == 0:
            ODE_i = m[i] * x_tt[i] + (mu[i] + mu[i + 1]) * x_t[i] + (omega[i] + omega[i + 1]) * x[i] - omega[i + 1] * x[
                i + 1] - mu[i + 1] * x_t[i + 1]
        elif i == n - 1:
            ODE_i = m[i] * x_tt[i] + (mu[i] + mu[i + 1]) * x_t[i] + (omega[i] + omega[i + 1]) * x[i] - omega[i] * x[
                i - 1] - mu[i] * x_t[i - 1]
        else:
            ODE_i = m[i] * x_tt[i] + (mu[i] + mu[i + 1]) * x_t[i] + (omega[i] + omega[i + 1]) * x[i] - omega[i + 1] * x[
                i + 1] - mu[i + 1] * x_t[i + 1] - omega[i] * x[i - 1] - mu[i] * x_t[i - 1]

        ODE_output.append(ODE_i)


    loss = sum(torch.mean(torch.square(ODE_i)) for ODE_i in ODE_output)

    return loss

# Arrays to store the loss values
pde_losses = []
mse_losses = []
ic_losses = []


# Record the initial time
start_time = time.time()

for epoch in range(EPOCHS):
    for batch_idx, (x_train, y1_train, y2_train, v1_train, v2_train) in enumerate(loader):
        model.enforce_non_negative()
        model.train()
        optimizer.zero_grad()

        x_train = x_train.view(-1, 1).to(DEVICE)
        y1_train = y1_train.view(-1, 1).to(DEVICE)
        y2_train = y2_train.view(-1, 1).to(DEVICE)
        v1_train = v1_train.view(-1, 1).to(DEVICE)
        v2_train = v2_train.view(-1, 1).to(DEVICE)
        output = model(x_train)
        loss_1 = MSE_LOSS(output[:, 0].view(-1, 1), y1_train)
        loss_2 = MSE_LOSS(output[:, 1].view(-1, 1), y2_train)
        loss_MSE = (loss_1 + loss_2)

        loss_PDE = PDE_loss(2)

        initial_conditions = [output[0, 0] - 0, output[0, 1] - 0, output[0, 1] - 0, output[0, 1] - 0]
        loss_IC = sum(torch.mean(ic ** 2) for ic in initial_conditions)

        total_loss = Weight_PDE * loss_PDE + Weight_MSE * loss_MSE + Weight_IC * loss_IC

        # Check for NaN in total_loss
        if torch.isnan(total_loss):
            print(f"NaN detected in total_loss at epoch {epoch}, batch {batch_idx}")
            break

        total_loss.backward(retain_graph=True)

        optimizer.step()
        scheduler.step()

        # Log GPU utilization
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        gpu_utilizations.append(utilization)

        # Enforce non-negativity of parameters
        model.enforce_non_negative()

        pde_losses.append(loss_PDE.item())
        mse_losses.append(loss_MSE.item())
        ic_losses.append(loss_IC.item())

        optimizer_weights.step()

        # Clip gradients of loss weights
        torch.nn.utils.clip_grad_norm_([Weight_PDE, Weight_MSE, Weight_IC], max_norm=1.0)

        print(f"Epoch {epoch} Loss: {total_loss.item()}")

        # GradNorm Algorithm Implementation
        with torch.no_grad():
            # Normalize the loss weights (if needed)
            Weight_PDE.data = Weight_PDE.data / (Weight_PDE.data + Weight_MSE.data + Weight_IC.data)
            Weight_MSE.data = Weight_MSE.data / (Weight_PDE.data + Weight_MSE.data + Weight_IC.data)
            Weight_IC.data = Weight_IC.data / (Weight_PDE.data + Weight_MSE.data + Weight_IC.data)

            # Get the gradient norms for each task
            norm_PDE = torch.norm(Weight_PDE.grad)
            norm_MSE = torch.norm(Weight_MSE.grad)
            norm_IC = torch.norm(Weight_IC.grad)

            # Calculate the target norm for each task
            mean_norm = (norm_PDE + norm_MSE + norm_IC) / 3
            target_norm_PDE = mean_norm * (loss_PDE / total_loss).item()
            target_norm_MSE = mean_norm * (loss_MSE / total_loss).item()
            target_norm_IC = mean_norm * (loss_IC / total_loss).item()

            # Update the loss weights based on the target norms
            Weight_PDE.data = Weight_PDE.data * (norm_PDE / target_norm_PDE)
            Weight_MSE.data = Weight_MSE.data * (norm_MSE / target_norm_MSE)
            Weight_IC.data = Weight_IC.data * (norm_IC / target_norm_IC)



    if torch.isnan(total_loss):
        print(f"Training halted due to NaN in total_loss at epoch {epoch}")
        break

# Save the model
#torch.save(model, 'PINN_gradnorm.pkl')

# Record the final time
end_time = time.time()
total_time = end_time - start_time

# Print the total time taken
print(f"Total time taken: {total_time} seconds")

# Plot GPU utilization
plt.plot(gpu_utilizations)
plt.xlabel('Epoch')
plt.ylabel('GPU Utilization (%)')
plt.title('GPU Utilization over Time')
plt.show()

# At the end of the training, plot the loss values
plt.figure(figsize=(10, 6))
plt.plot(pde_losses, label='PDE Loss')
plt.plot(mse_losses, label='MSE Loss')
plt.plot(ic_losses, label='IC Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Losses over Epochs')
plt.grid(True)
plt.show()



#ODE Test from Model parameter
mu1 = model.mu1.item()
mu2 = model.mu2.item()
mu3 = model.mu3.item()
omega1 = model.omega1.item()
omega2 = model.omega2.item()
omega3 = model.omega3.item()

# Store them as a dictionary for easy access
trained_parameters = {
    'mu1': mu1,
    'mu2': mu2,
    'mu3': mu3,
    'omega1': omega1,
    'omega2': omega2,
    'omega3': omega3
}

# Print or use the trained parameters as needed
print(trained_parameters)

m1, m2 = 1.0, 1.0
t = np.linspace(0, 3, 300)
u1_model, u2_model, v1_model, v2_model = solve_spring_mass_damper_eigenvalue(m1, m2, omega1, omega2, omega3, mu1, mu2, mu3, t)

# Sampled ground truth data
x_ground_truth = torch.tensor(t[0:300:2], dtype=torch.float16)
y1_ground_truth = torch.tensor(u1_t[0:300:2], dtype=torch.float16)
y2_ground_truth = torch.tensor(u2_t[0:300:2], dtype=torch.float16)

plt.plot(t, u1_t, color='tab:orange', linestyle='--', label='x1')
plt.plot(t, u2_t, color='tab:blue', linestyle='--', label='x2')
plt.plot(t, u1_model, color='tab:orange', label='x1 from model predicted parameter')
plt.plot(t, u2_model, color='tab:blue', label='x2 from model predicted parameter')
plt.title(f'Final Prediction -- Learned d1: {model.mu1.item():.4f} -- Learned k1: {model.omega1.item():.4f} -- Learned d2: {model.mu2.item():.4f} -- Learned k2:{model.omega2.item():.4f}-- Learned d3: {model.mu3.item():.4f} -- Learned k3:{model.omega3.item():.4f}')
plt.legend(['Ground Truth x1', 'Ground Truth x2', 'x1 from model predicted parameter', 'x2 from model predicted parameter'])
plt.show()

plt.plot(t, v1_t, color='tab:orange', linestyle='--', label='v1')
plt.plot(t, v2_t, color='tab:blue', linestyle='--', label='v2')
plt.plot(t, v1_model, color='tab:orange', label='v1 from model predicted parameter')
plt.plot(t, v2_model, color='tab:blue', label='v2 from model predicted parameter')
plt.title(f'Final Prediction -- Learned d1: {model.mu1.item():.4f} -- Learned k1: {model.omega1.item():.4f} -- Learned d2: {model.mu2.item():.4f} -- Learned k2:{model.omega2.item():.4f}-- Learned d3: {model.mu3.item():.4f} -- Learned k3:{model.omega3.item():.4f}')
plt.legend(['Ground Truth v1', 'Ground Truth v2', 'v1 from model predicted parameter', 'v2 from model predicted parameter'])
plt.show()
