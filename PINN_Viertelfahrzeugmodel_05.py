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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize NVML
pynvml.nvmlInit()
device_index = 0  # Change this if you have multiple GPUs
handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)


def generate_parameter_combinations(c_F_values, d_F_values, c_R_values, d_R_values):
    return list(itertools.product(c_F_values, d_F_values, c_R_values, d_R_values))

def mass_spring_damper(state, t, m, d, k):
    x, v = state  # unpack the state vector
    dxdt = v  # derivative of x is velocity
    dvdt = (-d * v - k * x) / m  # derivative of v is acceleration
    return [dxdt, dvdt]

def solve_harmonic_oscillator(m, d, k, x0, v0, t):
    initial_state = [x0, v0]
    states = odeint(mass_spring_damper, initial_state, t, args=(m, d, k))
    return states[:, 0], states[:, 1]

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
        plt.scatter(xp.cpu(), -0 * torch.ones_like(xp.cpu()), s=60, color="tab:green", alpha=0.4,
                    label='Physics loss training locations')
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

def calculate_damping_quality_kpi(k, d, m):
    eigenfrequency = np.sqrt(k / m) / (2 * np.pi)  # Eigenfrequenz
    damping_ratio = d / (2 * np.sqrt(k * m))  # Dämpfungsgrad
    quality_factor = 1 / (2 * damping_ratio)  # Gütefaktor
    return eigenfrequency, damping_ratio, quality_factor

def adaptive_sampling(time, displacement, threshold=0.1, max_interval=0.1):
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

# Example harmonic oscillator data
def harmonic_oscillator(t, A=1, omega=1, phase=0):
    return A * np.sin(omega * t + phase)


# Main process
c_F_values = np.linspace(10000, 100000, 5)  # Example values for chassis stiffness
d_F_values = np.linspace(500, 10000, 5)  # Example values for chassis damping
c_R_values = np.linspace(100000, 500000, 5)  # Example values for tire stiffness
d_R_values = np.linspace(10, 250, 5) # Example values for tire damping

combinations = generate_parameter_combinations(c_F_values, d_F_values, c_R_values, d_R_values)

start = tm.time()

# for i, combination in enumerate(selected_combinations):
for i, combination in enumerate(combinations):
    # c_F, d_F, eigenfrequency, damping_ratio, quality_factor = combination
    c_F, d_F, c_R, d_R = combination
    # print(f"Training with parameters: c_F={c_F}, d_F={d_F}, eigenfrequency={eigenfrequency}, damping_ratio={damping_ratio}, quality_factor={quality_factor}")
    print(f"Dataset {i + 1}/{len(combinations)} in progress...")

    # The parameters of tyre are constant
    # c_R = 365834  # N/m, stiffness of the tire
    # d_R = 80  # Ns/m, damping of the tire

    print(
        f"c_F: {c_F}\td_F: {d_F}\t"
        f"c_R: {c_R}\td_R: {d_R}\t"
    )

    csv_file_path = 'Dataset_BMW_KPI_CurveFitting.csv'


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


    # Initial conditions: [z_BO, v_BO, z_T, v_T]
    y0 = [0.1, 0, 0.1, 0]

    # Time span
    t_span = (0, 6)  # simulate for 6 seconds
    t = np.linspace(0, 6, 600)

    # Parameters
    m = 1  # kg, mass of the body

    # Parameters
    m_F = 537  # kg, mass of the body
    m_R = 68  # kg, mass of the tire
    c_F = c_F  # N/m, stiffness of the chasis
    d_F = d_F  # Ns/m, damping of the chasis
    c_R = c_R  # N/m, stiffness of the tire
    d_R = d_R  # Ns/m, damping of the tire

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
    # plt.show()

    t = np.linspace(0, 6, 600)
    x = torch.tensor(t[0:600:1], dtype=torch.float32).view(-1, 1).to(device)
    y = torch.tensor(z_F[0:600:1], dtype=torch.float32).view(-1, 1).to(device)
    v = torch.tensor(v_F[0:600:1], dtype=torch.float32).view(-1, 1).to(device)

    # Estimate initial k and d
    k_1d, d_1d = estimate_k_d(t, y, m)
    print(f"Estimated k: {k_1d}")
    print(f"Estimated d: {d_1d}")


    # Calculate the damping quality KPI
    eigenfrequency, damping_ratio, quality_factor = calculate_damping_quality_kpi(k_1d, d_1d, m)

    print(
        f"Eigenfrequency: {eigenfrequency}\tDamping ratio: {damping_ratio}\t"
        f"Quality factor: {quality_factor}\t"
    )


    # Time array
    t = np.linspace(0, 6, 600)  # 10 seconds, 250 points

    # Initial conditions
    x0 = 0.1  # initial displacement in meters
    v0 = 0.0  # initial velocity in m/s

    # Solve
    x_t, v_t = solve_harmonic_oscillator(m, d_1d, k_1d, x0, v0, t)

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
    plt.tight_layout()
    file = "plots_pinn1d_Viertelfahrzeugmodel_Training_CurveFitting/training_result_pinn_%i.png" % (i + 1)
    plt.savefig(file)
    # plt.show()

    # Save the results to the CSV file
    results = [c_F, d_F, c_R, d_R, eigenfrequency, damping_ratio, quality_factor]
    save_results_to_csv(csv_file_path, results)


end = tm.time()
training_time = end - start
print(training_time)
