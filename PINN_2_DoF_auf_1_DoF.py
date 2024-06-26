import time

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.linalg import expm
from torch.utils.data import DataLoader, TensorDataset
import pynvml
import matplotlib.animation as animation

# Initialize NVML
pynvml.nvmlInit()
device_index = 0  # Change this if you have multiple GPUs
handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

#plt.style.use('dark_background')




# Function to safely convert a tensor to numpy array
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().numpy() if x.requires_grad else x.numpy()
    return x

def derivative(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]


def harmonic_oscillator_solution(m, d, k, t):
    # Calculate the discriminant
    delta = d ** 2 - 4 * m * k

    if delta > 0:
        # Overdamped case
        r1 = (-d + np.sqrt(delta)) / (2 * m)
        r2 = (-d - np.sqrt(delta)) / (2 * m)
        # General solution: x(t) = C1 * exp(r1 * t) + C2 * exp(r2 * t)
        # Assuming initial conditions x(0) = x0, x'(0) = v0
        x0, v0 = 0.5, 0  # example initial conditions
        A = np.array([[1, 1], [r1, r2]])
        B = np.array([x0, v0])
        C1, C2 = np.linalg.solve(A, B)
        x_t = C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t)

    elif delta == 0:
        # Critically damped case
        r = -d / (2 * m)
        # General solution: x(t) = (C1 + C2 * t) * exp(r * t)
        # Assuming initial conditions x(0) = x0, x'(0) = v0
        x0, v0 = 0.5, 0  # example initial conditions
        C1 = x0
        C2 = v0 - r * x0
        x_t = (C1 + C2 * t) * np.exp(r * t)

    else:
        # Underdamped case
        alpha = -d / (2 * m)
        beta = np.sqrt(4 * m * k - d ** 2) / (2 * m)
        # General solution: x(t) = exp(alpha * t) * (C1 * cos(beta * t) + C2 * sin(beta * t))
        # Assuming initial conditions x(0) = x0, x'(0) = v0
        x0, v0 = 0.5, 0  # example initial conditions
        C1 = x0
        C2 = (v0 - alpha * x0) / beta
        x_t = np.exp(alpha * t) * (C1 * np.cos(beta * t) + C2 * np.sin(beta * t))

    plt.plot(t, x_t)
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (x)')
    plt.title('Harmonic Oscillator Mass-Spring-Damper System')
    plt.grid(True)
    plt.show()

    return x_t


# Example usage:
m = 1.0  # mass
d = 0.6  # damping coefficient
k = 8.0  # spring constant
t = np.linspace(0, 6, 600)

x_t = harmonic_oscillator_solution(m, d, k, t)
x_t = torch.tensor(x_t[0:600:1], dtype=torch.float32, device=torch.device("cpu"))



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
    x0 = np.array([0.5, 0, 0.5, 0])  # initial_u1, initial_v1, initial_u2, initial_v2
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
t = np.linspace(0, 6, 600)

u1_t, u2_t, v1_t, v2_t = solve_spring_mass_damper_eigenvalue(m1, m2, k1, k2, k3, d1, d2, d3, t)

# Sampled ground truth data
x_ground_truth = torch.tensor(t[0:600:1], dtype=torch.float32, device=torch.device("cpu"))
y1_ground_truth = torch.tensor(u1_t[0:600:1], dtype=torch.float32, device=torch.device("cpu"))
y2_ground_truth = torch.tensor(u2_t[0:600:1], dtype=torch.float32, device=torch.device("cpu"))
v1_ground_truth = torch.tensor(v1_t[0:600:1], dtype=torch.float32, device=torch.device("cpu"))
v2_ground_truth = torch.tensor(v2_t[0:600:1], dtype=torch.float32, device=torch.device("cpu"))

# create DataLoader, then take one batch
loader = DataLoader(list(zip(x_ground_truth, y1_ground_truth, y2_ground_truth, v1_ground_truth, v2_ground_truth, x_t)),
                    shuffle=True, batch_size=5000, pin_memory=False)


plt.plot(t, u1_t, label='u1(t)')
plt.plot(t, u2_t, label='u2(t)')
plt.plot(x_ground_truth, y1_ground_truth, 'o')
plt.plot(x_ground_truth, y2_ground_truth, 'o')
plt.legend(['Ground Truth u1', 'Ground Truth u2', 'Sampled Data u1', 'Sampled Data u2'])
plt.show()


# Define the neural network architecture
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(1, 100),
            nn.Sigmoid(),
            nn.Linear(100, 100),
            nn.Sigmoid(),
            nn.Linear(100, 100),
            nn.Sigmoid(),
            nn.Linear(100, 100),
            nn.Sigmoid(),
            nn.Linear(100, 100),
            nn.Sigmoid(),
            nn.Linear(100, 1)
        )
        # Initialize parameters k and d as trainable parameters with constraints
        self.k = nn.Parameter(torch.tensor(8.0, dtype=torch.float32))
        self.d = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, t):
        return self.hidden(t)

    def get_params(self):
        # Ensure k and d remain positive and d is not larger than k
        k = torch.abs(self.k)
        d = torch.abs(self.d)
        return k, d

def loss_function(model, t, x2, v2, x0, m=1.0):
    """
    Computes the loss function for a physics-informed neural network.

    Parameters:
    - model: The neural network model.
    - t: Tensor of time values.
    - x2: Tensor of true displacement values.
    - v2: Tensor of true velocity values.
    - x0: Initial displacement.
    - v0: Initial velocity.
    - m: Mass parameter, default is 1.0.
    - weight_physics: Weight for the physics loss term.
    - weight_data: Weight for the data loss term.
    - weight_velocity: Weight for the velocity loss term.
    - weight_initial: Weight for the initial condition loss term.
    - weight_regularization: Weight for the regularization loss term.

    Returns:
    - Tuple containing total loss and individual loss components.
    """

    weight_physics = 1.0
    weight_data = 1.0
    weight_velocity = 1.0
    weight_initial = 0.0
    weight_regularization = 1e-6

    # Predicted displacement
    x2_pred = model(t)

    # Compute the first derivative of predicted displacement w.r.t. time
    x2_t = torch.autograd.grad(x2_pred, t, torch.ones_like(x2_pred), create_graph=True)[0]

    # Compute the second derivative of predicted displacement w.r.t. time
    x2_tt = torch.autograd.grad(x2_t, t, torch.ones_like(x2_t), create_graph=True)[0]

    # Get parameters from the model
    k, d = model.get_params()

    # Simplified system equation: m * x2_tt + d * x2_t + k * x2_pred = 0
    physics_loss = torch.mean((m * x2_tt + d * x2_t + k * x2_pred) ** 2)

    # Data loss: mean squared error between predicted and true displacements
    data_loss = torch.mean((x2_pred - x2) ** 2)

    # Velocity loss: mean squared error between predicted and true velocities
    velocity_loss = torch.mean((x2_t - v2) ** 2)

    # Initial condition loss: ensuring the model respects the initial conditions
    initial_condition_loss = torch.mean((x2_pred[0] - x0) ** 2)

    # Regularization to enforce physical constraints
    regularization_loss = (k - 10.0) ** 2 + (d - 0.1) ** 2

    # Total loss is a combination of weighted physics, data, velocity, initial condition, and regularization losses
    total_loss = (weight_physics * physics_loss +
                  weight_data * data_loss +
                  weight_velocity * velocity_loss +
                  weight_initial * initial_condition_loss +
                  weight_regularization * regularization_loss)

    return total_loss, physics_loss, data_loss, velocity_loss, initial_condition_loss, regularization_loss



EPOCHS = 15000
gpu_utilizations = []

# Define the initial and target learning rates
initial_lr = 1e-3
target_lr = 1e-4

# Define the number of epochs between each learning rate update
update_interval = 5000

# Calculate the gamma value for the scheduler
gamma = (target_lr / initial_lr) ** (1 / (20000 / update_interval))

#DEVICE = "cpu"
DEVICE = torch.device("cuda")
print(DEVICE)

#model = torch.load('PINN_EDAG_2DoF_auf_1_DoF.pkl')
#model = model.to(DEVICE)
model = PINN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=update_interval, gamma=gamma)
MSE_LOSS = nn.MSELoss().to(DEVICE)


PDE_POINTS = 600


# Lists to store loss history
pde_loss_history = []
regularization_loss_history = []
ground_truth_loss2_history = []
velocity_truth_loss_history = []
ic_loss_history = []
overall_loss_history = []

start = time.time()

model.train()

# Example initial conditions
x0 = 0.5  # initial displacement


for epoch in range(EPOCHS):
    for batch_idx, (x_ground_truth, y1_ground_truth, y2_ground_truth, v1_ground_truth, v2_ground_truth, x_t) in enumerate(loader):

        #model.enforce_non_negative()

        x_ground_truth = x_ground_truth.view(-1, 1).to(DEVICE)
        y2_ground_truth = y2_ground_truth.view(-1, 1).to(DEVICE)
        v2_ground_truth = v2_ground_truth.view(-1, 1).to(DEVICE)
        x_t = x_t.view(-1, 1).to(DEVICE)

        # Time tensor for the training step
        t = torch.linspace(0, 6, 600, requires_grad=True).view(-1, 1).to(DEVICE)

        total_loss, physics_loss, data_loss, velocity_loss, initial_condition_loss, regularization_loss = loss_function(
            model, t, y2_ground_truth, v2_ground_truth, x0)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Log GPU utilization
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        gpu_utilizations.append(utilization)

        overall_loss_history.append(total_loss.item())
        pde_loss_history.append(physics_loss.item())
        ground_truth_loss2_history.append(data_loss.item())
        velocity_truth_loss_history.append(velocity_loss.item())
        ic_loss_history.append(initial_condition_loss.item())
        regularization_loss_history.append(regularization_loss.item())

        # Print the loss values and learned parameters every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch: {epoch + 1}\tTotal Loss: {total_loss.item()}\tPhysics Loss: {physics_loss.item()}\t"
                f"Data Loss: {data_loss.item()}\tVelocity Loss: {velocity_loss.item()}\t"
                f"Initial Condition Loss: {initial_condition_loss.item()}\tRegularization Loss: {regularization_loss.item()}"
            )
            # Print the current learning rate every 100 epochs
            print("Current Learning Rate:", optimizer.param_groups[0]['lr'])
            k, d = model.get_params()
            print(f'Epoch {epoch}, Total Loss: {total_loss.item()}, k: {k.item()}, d: {d.item()}')




end = time.time()
training_time = end - start
print(training_time)


#model.eval()
torch.save(model, 'PINN_EDAG_2DoF_auf_1_DoF.pkl')
#model = torch.load('PINN_EDAG_2DoF_auf_1_DoF.pkl')

# Plot GPU utilization
plt.plot(gpu_utilizations)
plt.xlabel('Epoch')
plt.ylabel('GPU Utilization (%)')
plt.title('GPU Utilization over Time')
plt.show()



# Plot the loss history
plt.figure(figsize=(10, 6))
plt.plot(pde_loss_history, label='PDE Loss')
plt.plot(velocity_truth_loss_history, label='Velocity Loss')
plt.plot(ic_loss_history, label='IC Loss')
plt.plot(regularization_loss_history, label='Regularization Loss')
plt.plot(ground_truth_loss2_history, label='Ground Truth Loss 2')
plt.plot(overall_loss_history, label='Overall Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss History')
plt.legend()
plt.grid(True)
plt.show()

#test_x = torch.linspace(0, 3, 300).reshape(-1, 1)
#position = model(test_x.to(DEVICE)).detach()
#plt.plot(t, u1_t, color='tab:orange', linestyle='--')
#plt.plot(test_x.view(-1), position[:, 0].cpu().view(-1), label='u1')
#plt.plot(t, u2_t, color='tab:green', linestyle='--')
#plt.plot(test_x.view(-1), position[:, 1].cpu().view(-1), label='u2')
#plt.title(f'Final Prediction -- Learned μ1: {model.mu1.item():.4f} -- Learned ω1: {model.omega1.item():.4f} -- Learned μ2: {model.mu2.item():.4f} -- Learned ω2: {model.omega2.item():.4f}')
#plt.legend(["Expected u1", "Predicted u1", "Expected u2", "Predicted u2"])
#plt.show()

# Retrieve the learned parameters and move them to CPU
omega1, mu1 = model.get_params()
omega1 = omega1.cpu().detach().numpy()
mu1 = mu1.cpu().detach().numpy()

# Print or use the trained parameters as needed
trained_parameters = {
    'mu1': mu1,
    'omega1': omega1
}

print(trained_parameters)

m1 = 1.0
t = np.linspace(0, 6, 600)
u2_model = harmonic_oscillator_solution(m1, mu1, omega1, t)

# Sampled ground truth data
x_ground_truth = torch.tensor(t[0:600:1], dtype=torch.float16)
y1_ground_truth = torch.tensor(u1_t[0:600:1], dtype=torch.float16)
y2_ground_truth = torch.tensor(u2_t[0:600:1], dtype=torch.float16)

plt.plot(t, u2_t, color='tab:blue', linestyle='--', label='x2')
plt.plot(t, u2_model, color='tab:blue', label='x2 from model predicted parameter')
plt.title(f'Final Prediction -- Learned d1: {mu1:.4f} -- Learned k1: {omega1:.4f}')
plt.legend(['Ground Truth x2', 'x2 from model predicted parameter'])
plt.show()



