import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

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

m1, m2 = 1.0, 1.0
t = np.linspace(0, 3, 300)

# Generating data for multiple combinations of k and d
k_values = [(10.0, 50.0, 25.0), (50.0, 1.0, 2.0)]
d_values = [(2.0, 1.0, 3.0), (0.5, 0.5, 0.5)]

training_data = []

for k1, k2, k3 in k_values:
    for d1, d2, d3 in d_values:
        u1_t, u2_t, v1_t, v2_t = solve_spring_mass_damper_eigenvalue(m1, m2, k1, k2, k3, d1, d2, d3, t)
        training_data.append((t, k1, k2, k3, d1, d2, d3, u1_t, u2_t))

# Convert training data to tensors
train_tensors = []

for data in training_data:
    t, k1, k2, k3, d1, d2, d3, u1_t, u2_t = data
    train_tensors.append((
        torch.tensor(t, dtype=torch.float32),
        torch.tensor([k1, k2, k3], dtype=torch.float32),
        torch.tensor([d1, d2, d3], dtype=torch.float32),
        torch.tensor(u1_t, dtype=torch.float32),
        torch.tensor(u2_t, dtype=torch.float32)
    ))

# Sample data for plotting
t_sample, k_sample, d_sample, u1_sample, u2_sample = train_tensors[0]

plt.plot(t_sample, u1_sample, label='u1(t)')
plt.plot(t_sample, u2_sample, label='u2(t)')
plt.legend()
plt.show()


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
            nn.Linear(7, 128),  # 1 (time) + 3 (k values) + 3 (d values)
            nn.Sigmoid(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
            nn.Linear(128, 2),  # 2 output values: u1 and u2
        )

    def forward(self, t, k, d):
        x = torch.cat((t, k, d), dim=1)
        return self.model(x)

    def enforce_non_negative(self):
        self.mu1.data.clamp_(min=0)
        self.mu2.data.clamp_(min=0)
        self.mu3.data.clamp_(min=0)
        self.omega1.data.clamp_(min=0)
        self.omega2.data.clamp_(min=0)
        self.omega3.data.clamp_(min=0)


EPOCHS = 10000

initial_lr = 1e-2
target_lr = 1e-3
update_interval = 5000
gamma = (target_lr / initial_lr) ** (1 / (20000 / update_interval))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = PINN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=update_interval, gamma=gamma)
MSE_LOSS = nn.MSELoss()

Weight_PDE = 1e-4
Weight_MSE = 1
Weight_Velocity = 1e-4

PDE_POINTS = 400


def PDE_loss(n):
    mu = [getattr(model, f'mu{i}') for i in range(1, n + 2)]
    omega = [getattr(model, f'omega{i}') for i in range(1, n + 2)]
    m = [1.0, 1.0]

    t = torch.linspace(0, 3, PDE_POINTS, requires_grad=True).view(-1, 1).to(DEVICE)
    k = torch.tensor([10.0, 50.0, 25.0], dtype=torch.float32).to(DEVICE).view(1, -1).repeat(PDE_POINTS, 1)
    d = torch.tensor([2.0, 1.0, 3.0], dtype=torch.float32).to(DEVICE).view(1, -1).repeat(PDE_POINTS, 1)
    x = model(t, k, d)
    x = [x[:, i].view(-1, 1) for i in range(n)]

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


pde_loss_history = []
ground_truth_loss1_history = []
ground_truth_loss2_history = []
velocity_truth_loss_history = []
overall_loss_history = []

model.train()

for epoch in range(EPOCHS):
    optimizer.zero_grad()

    for data in train_tensors:
        t, k, d, y1_ground_truth, y2_ground_truth = [d.to(DEVICE) for d in data]

        t = t.view(-1, 1)
        k = k.view(1, -1).repeat(t.shape[0], 1)
        d = d.view(1, -1).repeat(t.shape[0], 1)

        pde_loss = PDE_loss(2)

        y_pred = model(t, k, d)
        y1_pred = y_pred[:, 0].view(-1, 1)
        y2_pred = y_pred[:, 1].view(-1, 1)
        v1_pred = derivative(y1_pred, t)
        v2_pred = derivative(y2_pred, t)

        ground_truth_loss1 = MSE_LOSS(y1_pred, y1_ground_truth.view(-1, 1))
        ground_truth_loss2 = MSE_LOSS(y2_pred, y2_ground_truth.view(-1, 1))
        velocity_loss = MSE_LOSS(v1_pred, derivative(y1_ground_truth.view(-1, 1), t)) + MSE_LOSS(v2_pred, derivative(
            y2_ground_truth.view(-1, 1), t))

        loss = (Weight_MSE) * (ground_truth_loss1 + ground_truth_loss2) \
               + (Weight_PDE) * pde_loss \
               + (Weight_Velocity) * velocity_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        model.enforce_non_negative()

        pde_loss_history.append(pde_loss.item())
        ground_truth_loss1_history.append(ground_truth_loss1.item())
        ground_truth_loss2_history.append(ground_truth_loss2.item())
        velocity_truth_loss_history.append(velocity_loss.item())
        overall_loss_history.append(loss.item())

    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{EPOCHS} - Loss: {loss.item()}')

# Plotting the loss history
plt.figure(figsize=(10, 8))
plt.plot(pde_loss_history, label='PDE Loss')
plt.plot(ground_truth_loss1_history, label='Ground Truth Loss 1')
plt.plot(ground_truth_loss2_history, label='Ground Truth Loss 2')
plt.plot(velocity_truth_loss_history, label='Velocity Truth Loss')
plt.plot(overall_loss_history, label='Overall Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss History')
plt.show()


#------------------------------------------------MODEL TESTING------------------------------------------------#


# Define new parameters for testing
k_new = (12.0, 48.0, 28.0)
d_new = (2.2, 1.2, 2.8)
t_test = np.linspace(0, 3, 300)

# Generate analytical solution for the new parameters
u1_test, u2_test, v1_test, v2_test = solve_spring_mass_damper_eigenvalue(m1, m2, *k_new, *d_new, t_test)

# Convert to tensors for plotting
t_test_tensor = torch.tensor(t_test, dtype=torch.float32).view(-1, 1).to(DEVICE)
k_test_tensor = torch.tensor(k_new, dtype=torch.float32).view(1, -1).repeat(t_test_tensor.shape[0], 1).to(DEVICE)
d_test_tensor = torch.tensor(d_new, dtype=torch.float32).view(1, -1).repeat(t_test_tensor.shape[0], 1).to(DEVICE)

# Predict displacements using the trained model
model.eval()
with torch.no_grad():
    y_pred_test = model(t_test_tensor, k_test_tensor, d_test_tensor)
    u1_pred_test = y_pred_test[:, 0].cpu().numpy()
    u2_pred_test = y_pred_test[:, 1].cpu().numpy()

# Plot analytical solutions and model predictions
plt.figure(figsize=(12, 6))

# Plot analytical solutions
plt.plot(t_test, u1_test, label='Analytical u1(t)', linestyle='--')
plt.plot(t_test, u2_test, label='Analytical u2(t)', linestyle='--')

# Plot model predictions
plt.plot(t_test, u1_pred_test, label='Model Prediction u1(t)', linestyle='-')
plt.plot(t_test, u2_pred_test, label='Model Prediction u2(t)', linestyle='-')

plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Comparison of Analytical and Predicted Displacements')
plt.legend()
plt.grid(True)
plt.show()
