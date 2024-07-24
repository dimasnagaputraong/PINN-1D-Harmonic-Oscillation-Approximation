from PIL import Image
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
from scipy.linalg import eig
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import pynvml
import csv

# Initialize NVML
pynvml.nvmlInit()
device_index = 0  # Change this if you have multiple GPUs
handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

csv_file_path = 'Dataset_BMW_KPI.csv'

def mass_spring_damper(state, t, m, d, k):

    x, v = state  # unpack the state vector
    dxdt = v  # derivative of x is velocity
    dvdt = (-d * v - k * x) / m  # derivative of v is acceleration
    return [dxdt, dvdt]

# Function to solve the harmonic oscillator
def solve_harmonic_oscillator(m, d, k, x0, v0, t):
    initial_state = [x0, v0]
    states = odeint(mass_spring_damper, initial_state, t, args=(m, d, k))

    # Extract position and velocity
    position = states[:, 0]
    velocity = states[:, 1]

    # Calculate acceleration
    acceleration = (-k * position - d * velocity) / m

    return position, velocity, acceleration


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

# Save results to CSV
def save_results_to_csv(file_path, results):
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(results)

def estimate_k_d(t, y, m):
    # Ensure y is a numpy array for easier manipulation
    y = y.cpu().numpy().flatten()

    def damped_oscillator(t, A, B, omega_n, zeta):
        if zeta < 1:
            omega_d = omega_n * np.sqrt(1 - zeta ** 2)
            return np.exp(-zeta * omega_n * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
        elif zeta == 1:
            return (A + B * t) * np.exp(-omega_n * t)
        else:
            term1 = np.exp(-zeta * omega_n * t)
            term2 = np.exp(omega_n * np.sqrt(zeta ** 2 - 1) * t)
            term3 = np.exp(-omega_n * np.sqrt(zeta ** 2 - 1) * t)
            return term1 * (A * term2 + B * term3)

    # Initial guesses for A, B, omega_n, and zeta
    initial_guess = [y[0], 0, 1.0, 0.5]

    # Fit the damped oscillator model to the data
    params, _ = curve_fit(damped_oscillator, t, y, p0=initial_guess)
    A, B, omega_n, zeta = params

    # Calculate initial estimates for k and d
    k = m * omega_n ** 2
    d = 2 * zeta * np.sqrt(k * m)

    return k, d

def oscillator(m, k, d, t):
    """Defines the analytical solution to the 1D harmonic oscillator problem with zero initial displacement."""
    w0 = np.sqrt(k / m)
    zeta = d / (2 * np.sqrt(k * m))  # Damping ratio

    if zeta < 1:  # Underdamped
        w_d = w0 * np.sqrt(1 - zeta**2)
        A = 1  # Initial amplitude
        phi = 0  # Phase
        y = A * torch.exp(-zeta * w0 * t) * torch.cos(w_d * t + phi)

    elif zeta == 1:  # Critically damped
        A = 1  # Initial amplitude
        B = 0  # Assume no initial velocity for simplicity
        y = (A + B * t) * torch.exp(-w0 * t)

    else:  # Overdamped
        r1 = -w0 * (zeta + np.sqrt(zeta**2 - 1))
        r2 = -w0 * (zeta - np.sqrt(zeta**2 - 1))
        A = 0.5  # Initial amplitude
        B = 0.5  # Assume no initial velocity for simplicity
        y = A * torch.exp(r1 * t) + B * torch.exp(r2 * t)

    return y

def calculate_damping_quality_kpi(k, d, m):
    eigenfrequency = np.sqrt(k / m) / (2 * np.pi) #Eigenfrequenz
    damping_ratio = d / (2 * np.sqrt(k * m)) #Dämpfungsgrad
    quality_factor = 1 / (2 * damping_ratio) #Gütefaktor
    return eigenfrequency, damping_ratio, quality_factor

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

# Define the ODE system
def quarter_car_model_2(t, y):
    z_F, v_F, z_R, v_R = y
    dz_F_dt = v_F
    dz_R_dt = v_R
    dv_F_dt = (-c_F_2 * (z_F - z_R) - d_F_2 * (v_F - v_R)) / m_F
    dv_R_dt = (c_F_2 * (z_F - z_R) + d_F_2 * (v_F - v_R) - c_R_2 * (z_R - z_S(t)) - d_R_2 * (v_R - z_S_dot(t))) / m_R
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

# Adaptive sampling function
def adaptive_sampling(time, displacement, threshold=0.1, max_interval=0.05):
    adaptive_time = [time[0]]
    adaptive_displacement = [displacement[0]]
    last_sample_time = time[0]

    for i in range(1, len(time) - 1):
        rate_of_change = abs(displacement[i + 1] - displacement[i - 1]) / (time[i + 1] - time[i - 1])
        if rate_of_change > threshold or (time[i] - last_sample_time) >= max_interval:
            adaptive_time.append(time[i])
            adaptive_displacement.append(displacement[i])
            last_sample_time = time[i]

    adaptive_time.append(time[-1])
    adaptive_displacement.append(displacement[-1])

    return np.array(adaptive_time), np.array(adaptive_displacement)



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Parameters
m_F = 537  # kg, mass of the body
m_R = 68    # kg, mass of the tire
c_F = 41332  # N/m, stiffness of the chasis
d_F = 1224   # Ns/m, damping of the chasis
c_R = 365834  # N/m, stiffness of the tire
d_R = 80    # Ns/m, damping of the tire

c_F_2 = 14596.12078672   # N/m, stiffness of the chasis
d_F_2 = 6861.11495938  # Ns/m, damping of the chasis
c_R_2 = 289037.31772931  # N/m, stiffness of the tire
d_R_2 = 422.44128236    # Ns/m, damping of the tire




# Initial conditions: [z_BO, v_BO, z_T, v_T]
y0 = [0.1, 0, 0.1, 0]

# Time span
t_span = (0, 6)  # simulate for 6 seconds
t = np.linspace(0, 6, 600)

# Solve the ODE
sol = solve_ivp(quarter_car_model, t_span, y0, t_eval=t)
sol_2 = solve_ivp(quarter_car_model_2, t_span, y0, t_eval=t)

# Extract the solution
z_F = sol.y[0]
v_F = sol.y[1]
z_R = sol.y[2]
v_R = sol.y[3]
t = sol.t

# Extract the solution
z_F_2 = sol_2.y[0]
v_F_2 = sol_2.y[1]
z_R_2 = sol_2.y[2]
v_R_2 = sol_2.y[3]
t = sol.t

# Compute accelerations
a_F = np.gradient(v_F, t)
a_R = np.gradient(v_R, t)

# Compute accelerations
a_F_2 = np.gradient(v_F_2, t)
a_R_2 = np.gradient(v_R_2, t)

# Compute the step input
z_S_vals = np.array([z_S(ti) for ti in t])

# Plotting
plt.figure(figsize=(12, 10))

# Plot road input and displacements in the same figure
plt.subplot(3, 1, 1)
plt.plot(t, z_S_vals, color='lightgrey', label='Road Input $z_S(t)$')
plt.plot(t, z_F, color='red', label='$z_{F}(t)$ - Displacement of Car Body BMW', linewidth=2)
plt.plot(t, z_R, color='lightgrey', label='$z_R(t)$ - Displacement of Tire BMW')
plt.plot(t, z_F_2, color='blue', label='$z_{F}(t)$ - Displacement of Car Body Opt', linewidth=2)
plt.plot(t, z_R_2, color='lightgrey', label='$z_R(t)$ - Displacement of Tire Opt')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.title('Road Input and Displacement Responses')
plt.legend()
plt.grid(True)

# Plot velocities
plt.subplot(3, 1, 2)
plt.plot(t, v_F, color='red',label='$v_{F}(t)$ - Velocity of Car Body BMW')
plt.plot(t, v_R, color='lightgrey',label='$v_R(t)$ - Velocity of Tire BMW')
plt.plot(t, v_F_2,color='blue', label='$v_{F}(t)$ - Velocity of Car Body Opt')
plt.plot(t, v_R_2, color='lightgrey',label='$v_R(t)$ - Velocity of Tire Opt')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity Responses')
plt.ylim((-2,2))
plt.legend()
plt.grid(True)

# Plot accelerations
plt.subplot(3, 1, 3)
plt.plot(t, a_F,color='red', label='$a_{F}(t)$ - Acceleration of Car Body BMW')
plt.plot(t, a_R, color='lightgrey',label='$a_R(t)$ - Acceleration of Tire BMW')
plt.plot(t, a_F_2,color='blue', label='$a_{F}(t)$ - Acceleration of Car Body Opt')
plt.plot(t, a_R_2, color='lightgrey',label='$a_R(t)$ - Acceleration of Tire Opt')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.title('Acceleration Responses')
plt.ylim((-50,50))
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


# Apply adaptive sampling
adaptive_time, adaptive_displacement = adaptive_sampling(t, z_F, threshold=0.05)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(t, z_F, label='Original Data')
plt.scatter(adaptive_time, adaptive_displacement, color='red', label='Adaptive Sampling', s=10)
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Adaptive Sampling of Harmonic Oscillator')
plt.legend()
plt.show()

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

#x_physics = torch.linspace(0, 6, 60).view(-1, 1).to(device).requires_grad_(True)  # Sample locations over the problem domain
x_physics = torch.tensor(adaptive_time, requires_grad=True, dtype=torch.float32).view(-1,1).to(device)


# Define the hyperparameter search space
space = [
    Real(1e-5, 1e-2, "log-uniform", name='initial_lr'),
    Integer(50, 500, name='model_neurons'),
    Integer(1, 5, name='model_layers')
]

# Define the objective function for Bayesian Optimization
@use_named_args(space)
def objective(**params):
    initial_lr = params['initial_lr']
    model_neurons = params['model_neurons']
    model_layers = params['model_layers']

    # Initialize the model with the given hyperparameters
    model = FCN(1, 1, model_neurons, model_layers, init_k, init_d).to(device)
    optimizer = optim.Adam([
        {'params': [param for name, param in model.named_parameters() if name not in ['k', 'd']]},
        {'params': [model.k, model.d]}], lr=initial_lr)

    # Training loop for a limited number of epochs to evaluate performance
    EPOCH = 1000  # Shortened for optimization purpose
    total_loss_history = []

    for i in range(EPOCH):
        optimizer.zero_grad()

        yh = model(x_data)
        loss1 = torch.mean((yh - y_data) ** 2)

        yhp = model(x_physics)
        dx = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0]
        dx2 = torch.autograd.grad(dx, x_physics, torch.ones_like(dx), create_graph=True)[0]
        physics = dx2 + (model.d / m) * dx + (model.k / m) * yhp
        loss2 = torch.mean(physics ** 2)

        loss = loss1 + (1e-4) * loss2
        loss.backward()
        optimizer.step()
        model.enforce_non_negative()

        total_loss_history.append(loss.item())

    final_loss = total_loss_history[-1]
    print(f"Params: {params}, Final Loss: {final_loss}")

    return final_loss



# Perform Bayesian Optimization
start_time = time.time()
result = gp_minimize(objective, space, n_calls=20, random_state=123)
end_time = time.time()

# Output results
print(f"Best parameters: {result.x}")
print(f"Best loss: {result.fun}")
print(f"Optimization time: {end_time - start_time} seconds")


torch.manual_seed(123)
model = FCN(1, 1, 300, 3, init_k, init_d).to(device)
initial_lr = 1e-3
#target_lr = 1e-4
#update_interval = 5000
#gamma = (target_lr / initial_lr) ** (1 / (30000 / update_interval))
optimizer = torch.optim.Adam([
    {'params': [param for name, param in model.named_parameters() if name not in ['k', 'd']]},
    {'params': [model.k, model.d]}], lr=initial_lr)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=update_interval, gamma=gamma)
EPOCH = 20000


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

    #vh = torch.autograd.grad(yh, x_data, torch.ones_like(yh), create_graph=True)[0]  # Computes dy/dx (velocity)
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


# Initial conditions
x0 = 0.1  # initial displacement in meters
v0 = 0.0  # initial velocity in m/s

# Time array
t = np.linspace(0, 6, 600)  # 10 seconds, 250 points

# Solve
x_t, v_t, a_t = solve_harmonic_oscillator(m, learned_d, learned_k, x0, v0, t)

rms_acceleration = np.sqrt(np.mean(a_t ** 2))

print(rms_acceleration)

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(x.cpu(), y.cpu(), color="grey", linewidth=2, alpha=0.8, label="Exact solution")
plt.plot(t, x_t, 'b', label='Displacement (x)')
plt.title('Displacement and Velocity Over Time')
plt.ylabel('Displacement (m)')
plt.legend(loc='best')

plt.subplot(212)
plt.plot(x.cpu(), v.cpu(), color="grey", linewidth=2, alpha=0.8, label="Exact solution")
plt.plot(t, v_t, 'r', label='Velocity (v)')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend(loc='best')
file = "plots_pinn1d_Viertelfahrzeugmodel/pinn_parameter_result.png"
plt.savefig(file)
plt.tight_layout()
plt.show()

#plt.figure(figsize=(8, 4))
#plt.plot(x.cpu(), y.cpu(), color="grey", linewidth=2, alpha=0.8, label="Exact solution")
#plt.plot(x.cpu(), y_learned.cpu(), color="tab:red", linewidth=2, alpha=0.8, label="Learned solution")
#plt.scatter(x_data.cpu(), y_data.cpu(), s=60, color="tab:orange", alpha=0.4, label='Training data')
#plt.xlabel('Time (s)')
#plt.ylabel('Displacement (m)')
#plt.title('Comparison of Exact and Learned Solutions')
#plt.legend()
#plt.grid(True)
#file = "plots_pinn1d_Viertelfahrzeugmodel/pinn_parameter_result.png"
#plt.savefig(file)
#plt.show()