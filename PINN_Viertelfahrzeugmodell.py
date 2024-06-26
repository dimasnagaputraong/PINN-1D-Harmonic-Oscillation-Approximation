import time
import sympy as sp
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.linalg import expm
from torch.utils.data import DataLoader
import pynvml

import matplotlib.animation as animation

# Initialize NVML
pynvml.nvmlInit()
device_index = 0  # Change this if you have multiple GPUs
handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)


def solve_quarter_car_eigenvalue(m_F, m_R, c_F, d_F, c_R, d_R, t):
    A = np.array([
        [0, 1, 0, 0],
        [-c_F / m_F, -d_F / m_F, c_F / m_F, d_F / m_F],
        [0, 0, 0, 1],
        [c_F / m_R, d_F / m_R, -(c_F + c_R) / m_R, -(d_F + d_R) / m_R]
    ])
    eigenvalues, eigenvectors = np.linalg.eig(A)
    Lambda = np.diag(eigenvalues)
    exp_Lambda_t = np.array([expm(Lambda * ti) for ti in t])
    V = eigenvectors
    V_inv = np.linalg.inv(V)
    x0 = np.array([0.5, 0, 0.5, 0])  # initial conditions [z_F, v_F, z_R, v_R]
    x_t = np.zeros((len(t), 4))
    for i in range(len(t)):
        x_t[i, :] = np.dot(np.dot(V, exp_Lambda_t[i]), V_inv @ x0)

    z_F = x_t[:, 0]
    v_F = x_t[:, 1]
    z_R = x_t[:, 2]
    v_R = x_t[:, 3]
    a_F = np.gradient(v_F, t)
    a_R = np.gradient(v_R, t)

    fig, axs = plt.subplots(3, 1, figsize=(10, 18))

    axs[0].plot(t, z_F, label='z_F(t)')
    axs[0].plot(t, z_R, label='z_R(t)')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Displacement')
    axs[0].set_title('Displacement over Time')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(t, v_F, label='v_F(t)')
    axs[1].plot(t, v_R, label='v_R(t)')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Velocity')
    axs[1].set_title('Velocity over Time')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(t, a_F, label='a_F(t)')
    axs[2].plot(t, a_R, label='a_R(t)')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Acceleration')
    axs[2].set_title('Acceleration over Time')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    return z_F, z_R, v_F, v_R


m_F = 537  # mass of Fahrwerk in kg
m_R = 68   # mass of Rad in kg
c_F = 41332  # suspension spring constant in N/m
d_F = 1224  # suspension damping constant in Ns/m
c_R = 365834  # tire spring constant in N/m
d_R = 80      # tire damping constant in Ns/m

t = np.linspace(0, 6, 600)  # 10 seconds

z_F, z_R, v_F, v_R = solve_quarter_car_eigenvalue(m_F, m_R, c_F, d_F, c_R, d_R, t)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().numpy() if x.requires_grad else x.numpy()
    return x

def derivative(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]

def squared_difference(input, target):
    return (input - target) ** 2

def harmonic_oscillator_solution(m, d, k, t):
    delta = d ** 2 - 4 * m * k

    if delta > 0:
        # Overdamped case
        r1 = (-d + np.sqrt(delta)) / (2 * m)
        r2 = (-d - np.sqrt(delta)) / (2 * m)
        x0, v0 = 0.5, 0  # example initial conditions
        A = np.array([[1, 1], [r1, r2]], dtype=np.float64)
        B = np.array([x0, v0], dtype=np.float64)
        C1, C2 = np.linalg.solve(A, B)
        x_t = C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t)

    elif delta == 0:
        # Critically damped case
        r = -d / (2 * m)
        x0, v0 = 0.5, 0  # example initial conditions
        C1 = x0
        C2 = v0 - r * x0
        x_t = (C1 + C2 * t) * np.exp(r * t)

    else:
        # Underdamped case
        alpha = -d / (2 * m)
        beta = np.sqrt(4 * m * k - d ** 2) / (2 * m)
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
m = 537  # mass
d = 1000  # damping coefficient
k = 40000  # spring constant
t = np.linspace(0, 6, 600)


x_ground_truth = torch.tensor(t[0:600:1], dtype=torch.float32, device=torch.device("cpu"))
y1_ground_truth = torch.tensor(z_R[0:600:1], dtype=torch.float32, device=torch.device("cpu"))
y2_ground_truth = torch.tensor(z_F[0:600:1], dtype=torch.float32, device=torch.device("cpu"))
v1_ground_truth = torch.tensor(v_R[0:600:1], dtype=torch.float32, device=torch.device("cpu"))
v2_ground_truth = torch.tensor(v_F[0:600:1], dtype=torch.float32, device=torch.device("cpu"))

loader = DataLoader(list(zip(x_ground_truth, y1_ground_truth, y2_ground_truth, v1_ground_truth, v2_ground_truth)),
                    shuffle=True, batch_size=600, pin_memory=False)

plt.plot(t, z_R, label='u1(t)')
plt.plot(t, z_F, label='u2(t)')
plt.plot(x_ground_truth, y1_ground_truth, 'o')
plt.plot(x_ground_truth, y2_ground_truth, 'o')
plt.legend(['Ground Truth u1', 'Ground Truth u2', 'Sampled Data u1', 'Sampled Data u2'])
plt.show()


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(1, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.k = torch.nn.Parameter(torch.FloatTensor([10000.0]), requires_grad=True)
        self.d = torch.nn.Parameter(torch.FloatTensor([1000.0]), requires_grad=True)

    def forward(self, t):
        return self.hidden(t)

    def get_params(self):
        k = torch.abs(self.k)
        d = torch.abs(self.d)
        return k, d

EPOCHS = 25000
gpu_utilizations = []

initial_lr = 1e-2
target_lr = 1e-3

update_interval = 5000

gamma = (target_lr / initial_lr) ** (1 / (15000 / update_interval))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

model = PINN().to(DEVICE)
optimizer = torch.optim.Adam([
    {'params': model.hidden.parameters(), 'lr': initial_lr},
    {'params': model.k, 'lr': initial_lr * 1000},
    {'params': model.d, 'lr': initial_lr * 1000}
])

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=update_interval, gamma=gamma)
MSE_LOSS = nn.MSELoss().to(DEVICE)

def loss_function(model, t, x2, v2, x0, m=537.0):
    weight_physics = 1e-3
    weight_data = 100.0
    weight_velocity = 1e-3
    weight_initial = 0.0

    x2_pred = model(t)
    x2_t = torch.autograd.grad(x2_pred.sum(), t, create_graph=True)[0]
    x2_tt = torch.autograd.grad(x2_t.sum(), t, create_graph=True)[0]

    k, d = model.get_params()


    ode = (m * x2_tt) + (d * x2_t) + (k * x2_pred)

    physics_loss = torch.mean(squared_difference(ode, torch.zeros_like(t)))
    data_loss = torch.mean(squared_difference(x2_pred, x2))
    velocity_loss = torch.mean(squared_difference(x2_t, v2))
    initial_condition_loss = torch.mean((x2_pred[0] - x0) ** 2)

    total_loss = (weight_physics * physics_loss +
                  weight_data * data_loss +
                  weight_velocity * velocity_loss +
                  weight_initial * initial_condition_loss)


    return total_loss, physics_loss, data_loss, velocity_loss, initial_condition_loss

PDE_POINTS = 600

pde_loss_history = []
ground_truth_loss2_history = []
velocity_truth_loss_history = []
ic_loss_history = []
overall_loss_history = []

start = time.time()

model.train()

x0 = 0.5  # initial displacement

for epoch in range(EPOCHS):
    for batch_idx, (x_ground_truth, y1_ground_truth, y2_ground_truth, v1_ground_truth, v2_ground_truth) in enumerate(loader):
        x_ground_truth = x_ground_truth.view(-1, 1).to(DEVICE)
        y2_ground_truth = y2_ground_truth.view(-1, 1).to(DEVICE)
        v2_ground_truth = v2_ground_truth.view(-1, 1).to(DEVICE)
        t = torch.linspace(0, 6, 600, requires_grad=True).view(-1, 1).to(DEVICE)

        total_loss, physics_loss, data_loss, velocity_loss, initial_condition_loss = loss_function(
            model, t, y2_ground_truth, v2_ground_truth, x0, m=537.0)

        optimizer.zero_grad()
        physics_loss.backward()

        # Print the gradient norms for each parameter every 100 epochs
        if (epoch + 1) % 100 == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"Parameter: {name}, Grad Norm: {param.grad.norm()}")

        # Manually scale the gradients for k and d
        with torch.no_grad():
            model.k.grad *= 1000
            model.d.grad *= 1000


        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)

        optimizer.step()
        scheduler.step()

        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        gpu_utilizations.append(utilization)


        overall_loss_history.append(total_loss.item())
        pde_loss_history.append(physics_loss.item())
        ground_truth_loss2_history.append(data_loss.item())
        velocity_truth_loss_history.append(velocity_loss.item())
        ic_loss_history.append(initial_condition_loss.item())


        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch: {epoch + 1}\tTotal Loss: {total_loss.item()}\tPhysics Loss: {physics_loss.item()}\t"
                f"Data Loss: {data_loss.item()}\tVelocity Loss: {velocity_loss.item()}\t"
                f"Initial Condition Loss: {initial_condition_loss.item()}"
            )
            print("Current Learning Rate:", optimizer.param_groups[0]['lr'])
            k, d = model.get_params()
            print(f'Epoch {epoch}, Total Loss: {total_loss.item()}, k: {k.item()}, d: {d.item()}')

end = time.time()
training_time = end - start
print(training_time)

torch.save(model, 'PINN_EDAG_Viertelfahrzeugmodell.pkl')

plt.plot(gpu_utilizations)
plt.xlabel('Epoch')
plt.ylabel('GPU Utilization (%)')
plt.title('GPU Utilization over Time')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(pde_loss_history, label='PDE Loss')
plt.plot(velocity_truth_loss_history, label='Velocity Loss')
plt.plot(ic_loss_history, label='IC Loss')
plt.plot(ground_truth_loss2_history, label='Ground Truth Loss 2')
plt.plot(overall_loss_history, label='Overall Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss History')
plt.legend()
plt.grid(True)
plt.show()

omega1, mu1 = model.get_params()
omega1 = omega1.cpu().detach().numpy()
mu1 = mu1.cpu().detach().numpy()

trained_parameters = {
    'mu1': mu1,
    'omega1': omega1
}

print(trained_parameters)

m1 = 537
t = np.linspace(0, 6, 600)
u2_model = harmonic_oscillator_solution(m1, mu1, omega1, t)

x_ground_truth = torch.tensor(t[0:600:1], dtype=torch.float16)
y2_ground_truth = torch.tensor(z_F[0:600:1], dtype=torch.float16)

plt.plot(t, z_F, color='tab:blue', linestyle='--', label='x2')
plt.plot(t, u2_model, color='tab:blue', label='x2 from model predicted parameter')
plt.legend(['Ground Truth x2', 'x2 from model predicted parameter'])
plt.show()
