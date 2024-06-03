import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
from scipy.linalg import expm


def derivative(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]

def solve_spring_mass_damper_eigenvalue(m1, m2, k1, k2, k3, d1, d2, d3, t):
    A = np.array([
        [0, 1, 0, 0],
        [-(k1 + k2) / m1, -(d1 + d2) / m1, k2 / m1, d2 / m1],
        [0, 0, 0, 1],
        [k2 / m2, d2 / m2, -(k2 + k3) / m2, -(d2 + d3) / m2]
    ])
    eigenvalues, eigenvectors = np.linalg.eig(A)
    Lambda = np.diag(eigenvalues)
    exp_Lambda_t = np.array([expm(Lambda * ti) for ti in t])
    V = eigenvectors
    V_inv = np.linalg.inv(V)
    x0 = np.array([0, 10, 0, -10])
    x_t = np.array([np.dot(np.dot(V, exp_Lambda_t[i]), np.dot(V_inv, x0)) for i in range(len(t))])

    u1_t = x_t[:, 0]
    u2_t = x_t[:, 2]
    v1_t = x_t[:, 1]
    v2_t = x_t[:, 3]
    a1_t = np.gradient(v1_t, t)
    a2_t = np.gradient(v2_t, t)

    return u1_t, u2_t

# Parameters for generating the ground truth data
m1, m2 = 1.0, 1.0
k1, k2, k3 = 10.0, 50.0, 25.0
d1, d2, d3 = 2.0, 1.0, 3.0
t = np.linspace(0, 3, 300)

u1_t, u2_t = solve_spring_mass_damper_eigenvalue(m1, m2, k1, k2, k3, d1, d2, d3, t)

x_ground_truth = torch.tensor(t[0:300:2], dtype=torch.float32)
y1_ground_truth = torch.tensor(u1_t[0:300:2], dtype=torch.float32)
y2_ground_truth = torch.tensor(u2_t[0:300:2], dtype=torch.float32)

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
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.model(x)

EPOCHS = 10000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 1e-3

model = PINN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
MSE_LOSS = nn.MSELoss()

PDE_POINTS = 300

def PDE_loss():
    mu1, mu2, mu3 = model.mu1, model.mu2, model.mu3
    omega1, omega2, omega3 = model.omega1, model.omega2, model.omega3
    m1, m2 = 1.0, 1.0

    t = torch.linspace(0, 3, PDE_POINTS, requires_grad=True).view(-1, 1).to(DEVICE)
    x = model(t)
    x1, x2 = x[:, 0].view(-1, 1), x[:, 1].view(-1, 1)

    x1_t, x2_t = derivative(x1, t), derivative(x2, t)
    x1_tt, x2_tt = derivative(x1_t, t), derivative(x2_t, t)

    ODE_output1 = m1 * x1_tt + (mu1+mu2) * x1_t + (omega1+omega2) * x1 - omega2 * x2 - mu2 * x2_t
    ODE_output2 = m2 * x2_tt + (mu2+mu3) * x2_t + (omega2+omega3) * x2 - omega2 * x1 - mu2 * x1_t

    loss = torch.mean(torch.square(ODE_output1)) + torch.mean(torch.square(ODE_output2))

    return loss

# Compute initial losses for normalization
with torch.no_grad():
    initial_pde_loss = PDE_loss().item()
    initial_ground_truth_loss1 = MSE_LOSS(model(x_ground_truth.to(DEVICE).view(-1, 1))[:, 0].view(-1, 1), y1_ground_truth.to(DEVICE).view(-1, 1)).item()
    initial_ground_truth_loss2 = MSE_LOSS(model(x_ground_truth.to(DEVICE).view(-1, 1))[:, 1].view(-1, 1), y2_ground_truth.to(DEVICE).view(-1, 1)).item()

# Lists to store loss history
pde_loss_history = []
ground_truth_loss1_history = []
ground_truth_loss2_history = []
overall_loss_history = []

model.train()

for epoch in range(EPOCHS):
    optimizer.zero_grad()

    pde_loss = PDE_loss()
    ground_truth_loss1 = MSE_LOSS(model(x_ground_truth.to(DEVICE).view(-1, 1))[:, 0].view(-1, 1), y1_ground_truth.to(DEVICE).view(-1, 1))
    ground_truth_loss2 = MSE_LOSS(model(x_ground_truth.to(DEVICE).view(-1, 1))[:, 1].view(-1, 1), y2_ground_truth.to(DEVICE).view(-1, 1))

    # Normalize losses
    normalized_pde_loss = pde_loss / (initial_pde_loss + 1e-8)
    normalized_ground_truth_loss1 = ground_truth_loss1 / (initial_ground_truth_loss1 + 1e-8)
    normalized_ground_truth_loss2 = ground_truth_loss2 / (initial_ground_truth_loss2 + 1e-8)

    loss = normalized_pde_loss + normalized_ground_truth_loss1 + normalized_ground_truth_loss2

    loss.backward()
    optimizer.step()

    pde_loss_history.append(pde_loss.item())
    ground_truth_loss1_history.append(ground_truth_loss1.item())
    ground_truth_loss2_history.append(ground_truth_loss2.item())
    overall_loss_history.append(loss.item())

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch: {epoch + 1}\tOverall Loss: {loss.item()}\tPDE Loss: {normalized_pde_loss.item()}\tGround Truth Loss1: {normalized_ground_truth_loss1.item()}\tGround Truth Loss2: {normalized_ground_truth_loss2.item()}"
              f"\tLearned d1: {model.mu1.item():.4f}\tLearned k1: {model.omega1.item():.4f}"
              f"\tLearned d2: {model.mu2.item():.4f}\tLearned k2: {model.omega2.item():.4f}"
              f"\tLearned d3: {model.mu3.item():.4f}\tLearned k3: {model.omega3.item():.4f}")

torch.save(model, 'PINN_EDAG_DN_2_Adam.pkl')

plt.figure(figsize=(10, 6))
plt.plot(pde_loss_history, label='PDE Loss')
plt.plot(ground_truth_loss1_history, label='Ground Truth Loss 1')
plt.plot(ground_truth_loss2_history, label='Ground Truth Loss 2')
plt.plot(overall_loss_history, label='Overall Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss History')
plt.legend()
plt.grid(True)
plt.show()

test_x = torch.linspace(0, 3, 300).reshape(-1, 1)
position = model(test_x.to(DEVICE)).detach()
plt.plot(t, u1_t, color='tab:orange', linestyle='--')
plt.plot(test_x.view(-1).cpu(), position[:, 0].cpu().view(-1), label='u1')
plt.plot(t, u2_t, color='tab:green', linestyle='--')
plt.plot(test_x.view(-1).cpu(), position[:, 1].cpu().view(-1), label='u2')
plt.title(f'Final Prediction -- Learned μ1: {model.mu1.item():.4f} -- Learned ω1: {model.omega1.item():.4f} -- Learned μ2: {model.mu2.item():.4f} -- Learned ω2: {model.omega2.item():.4f}')
plt.legend(["Expected u1", "Predicted u1", "Expected u2", "Predicted u2"])
plt.show()

mu1 = model.mu1.item()
mu2 = model.mu2.item()
mu3 = model.mu3.item()
omega1 = model.omega1.item()
omega2 = model.omega2.item()
omega3 = model.omega3.item()

trained_parameters = {
    'mu1': mu1,
    'mu2': mu2,
    'mu3': mu3,
    'omega1': omega1,
    'omega2': omega2,
    'omega3': omega3
}

print(trained_parameters)

m1, m2 = 1.0, 1.0
t = np.linspace(0, 3, 300)
u1_model, u2_model = solve_spring_mass_damper_eigenvalue(m1, m2, omega1, omega2, omega3, mu1, mu2, mu3, t)

plt.plot(t, u1_t, color='tab:orange', linestyle='--', label='u1')
plt.plot(t, u2_t, color='tab:green', linestyle='--', label='u2')
plt.plot(t, u1_model, label='u1_model')
plt.plot(t, u2_model, label='u2_model')
plt.legend(['Ground Truth u1', 'Ground Truth u2', 'Trained Data u1', 'Trained Data u2'])
plt.show()
