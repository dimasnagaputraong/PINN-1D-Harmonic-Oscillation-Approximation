import time

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.linalg import expm
from torch.utils.data import DataLoader, TensorDataset

#plt.style.use('dark_background')


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
loader = DataLoader(list(zip(x_ground_truth,y1_ground_truth,y2_ground_truth, v1_ground_truth, v2_ground_truth)), shuffle=True, batch_size=5000, pin_memory=True)



plt.plot(t, u1_t, label='u1(t)')
plt.plot(t, u2_t, label='u2(t)')
plt.plot(x_ground_truth, y1_ground_truth, 'o')
plt.plot(x_ground_truth, y2_ground_truth, 'o')
plt.legend(['Ground Truth u1', 'Ground Truth u2', 'Sampled Data u1', 'Sampled Data u2'])
plt.show()



print(y1_ground_truth.is_cuda)


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
            #nn.Linear(128, 128),
            #nn.Sigmoid(),
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

EPOCHS = 2000

# Define the initial and target learning rates
initial_lr = 1e-2
target_lr = 1e-3

# Define the number of epochs between each learning rate update
update_interval = 5000

# Calculate the gamma value for the scheduler
gamma = (target_lr / initial_lr) ** (1 / (20000 / update_interval))

#DEVICE = "cpu"
DEVICE = torch.device("cuda")
print(DEVICE)

#model = torch.load('PINN_EDAG_DN.pkl')
model = PINN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=update_interval, gamma=gamma)
MSE_LOSS = nn.MSELoss()

#Assign different weight for loss function
Weight_PDE = 1
Weight_MSE = 1
Weight_Velocity = 0
Weight_IC = 1

PDE_POINTS = 500

def PDE_loss(n):

    mu = [getattr(model, f'mu{i}') for i in range(1, n + 2)]
    omega = [getattr(model, f'omega{i}') for i in range(1, n + 2)]
    m = [1.0, 1.0]

    t = torch.linspace(0, 3, PDE_POINTS, requires_grad=True, device=torch.device("cuda")).view(-1, 1)
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

# Lists to store loss history
pde_loss_history = []
ground_truth_loss1_history = []
ground_truth_loss2_history = []
velocity_truth_loss_history = []
ic_loss_history = []
overall_loss_history = []

start = time.time()

model.train()

for epoch in range(EPOCHS):
    for x_ground_truth, y1_ground_truth, y2_ground_truth, v1_ground_truth, v2_ground_truth in loader:
        x_ground_truth, y1_ground_truth, y2_ground_truth, v1_ground_truth, v2_ground_truth = \
            x_ground_truth.to(DEVICE), y1_ground_truth.to(DEVICE), y2_ground_truth.to(DEVICE), v1_ground_truth.to(DEVICE), v2_ground_truth.to(DEVICE)

        print(x_ground_truth.is_cuda)

        if epoch % 1000 == 0:
            Weight_MSE = max(1e-4, Weight_MSE / 2)



        # Time tensor for the training step
        t = torch.linspace(0, 3, 300, requires_grad=True).view(-1, 1).to(DEVICE)

        # Define the initial conditions
        initial_t = torch.tensor([0.0], requires_grad=True).to(DEVICE)
        initial_v1 = torch.tensor([10.0], requires_grad=True).to(DEVICE)
        initial_v2 = torch.tensor([-10.0], requires_grad=True).to(DEVICE)
        initial_u1 = torch.tensor([0.0], requires_grad=True).to(DEVICE)
        initial_u2 = torch.tensor([0.0], requires_grad=True).to(DEVICE)

        # Compute PDE loss
        pde_loss = PDE_loss(2)

        # Model prediction for all time steps
        y_pred = model(t)
        y1_pred = y_pred[:, 0].view(-1, 1)
        y2_pred = y_pred[:, 1].view(-1, 1)

        # Compute derivatives
        v1_pred = derivative(y1_pred, t)
        v2_pred = derivative(y2_pred, t)

        # Compute ground truth loss
        ground_truth_loss1 = MSE_LOSS(model(x_ground_truth.to(DEVICE).view(-1, 1))[:, 0].view(-1, 1),
                                      y1_ground_truth.to(DEVICE).view(-1, 1))
        ground_truth_loss2 = MSE_LOSS(model(x_ground_truth.to(DEVICE).view(-1, 1))[:, 1].view(-1, 1),
                                      y2_ground_truth.to(DEVICE).view(-1, 1))



        # Initial conditions loss
        initial_y_pred = model(initial_t.view(-1, 1))

        # Ensure the initial_y_pred has the correct shape
        if initial_y_pred.dim() == 1:
            initial_y_pred = initial_y_pred.view(-1, 1)

        initial_v1_pred = derivative(initial_y_pred[:, 0].view(-1, 1), initial_t)
        initial_v2_pred = derivative(initial_y_pred[:, 1].view(-1, 1), initial_t)

        initial_u1_pred = initial_y_pred[:, 0].view(-1, 1)
        initial_u2_pred = initial_y_pred[:, 1].view(-1, 1)

        initial_conditions_loss = MSE_LOSS(initial_v1_pred, initial_v1.view(-1, 1)) + MSE_LOSS(initial_v2_pred,
                                                                                               initial_v2.view(-1, 1)) \
                                  + MSE_LOSS(initial_u1_pred, initial_u1.view(-1, 1)) + MSE_LOSS(initial_u2_pred,
                                                                                                 initial_u2.view(-1, 1))

        # Total loss
        loss = (Weight_MSE) * (ground_truth_loss1 + ground_truth_loss2) \
               + (Weight_PDE) * pde_loss \
               + (Weight_IC) * initial_conditions_loss  # Add initial conditions loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Enforce non-negativity of parameters
        model.enforce_non_negative()

        # Store the loss values
        pde_loss_history.append(pde_loss.item())
        ground_truth_loss1_history.append(ground_truth_loss1.item())
        ground_truth_loss2_history.append(ground_truth_loss2.item())
        #velocity_truth_loss_history.append(velocity_loss.item())
        ic_loss_history.append(initial_conditions_loss.item())
        overall_loss_history.append(loss.item())

        # Print the loss values and learned parameters every 1000 epochs
        if (epoch + 1) % 1000 == 0:
            print(
                f"Epoch: {epoch + 1}\tOverall Loss: {loss.item()}\tPDE Loss: {(1) * pde_loss.item()}\tGround Truth Loss1: {(1) * ground_truth_loss1.item()}"
                f"\tGround Truth Loss2: {ground_truth_loss2.item()}\tInitial Conditions Loss: {initial_conditions_loss.item()}"
                f"\tLearned d1: {model.mu1.item():.4f}\tLearned k1: {model.omega1.item():.4f}"
                f"\tLearned d2: {model.mu2.item():.4f}\tLearned k2: {model.omega2.item():.4f}"
                f"\tLearned d3: {model.mu3.item():.4f}\tLearned k3: {model.omega3.item():.4f}\tWeight MSE: {Weight_MSE:.4f}")
            # Print the current learning rate every 1000 epochs
            print("Current Learning Rate:", optimizer.param_groups[0]['lr'])



end = time.time()
training_time = end - start
print(training_time)

#model.eval()
#torch.save(model, 'PINN_EDAG_DN_2.pkl')
#model = torch.load('PINN_EDAG_DN.pkl')


# Plot the loss history
plt.figure(figsize=(10, 6))
plt.plot(pde_loss_history, label='PDE Loss')
plt.plot(velocity_truth_loss_history, label='Velocity Loss')
plt.plot(ic_loss_history, label='IC Loss')
plt.plot(ground_truth_loss1_history, label='Ground Truth Loss 1')
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