import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



class MaterialModelNN(nn.Module):
    #current architecture mimics the paper (6-32-6)
    def __init__(self, input_size, hidden_size, output_size):
        super(MaterialModelNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input to hidden layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, output_size)  # Hidden to output layer

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def get_effective_stiffness_matrix(self):
        # Extract the weights from both layers
        W1 = self.fc1.weight.data  # Shape: [hidden_size, input_size]
        W2 = self.fc2.weight.data  # Shape: [output_size, hidden_size]

        # Calculate the effective stiffness matrix C = W2 * W1
        stiffness_matrix = torch.matmul(W2, W1).cpu().numpy()  # Shape: [output_size, input_size]
        return stiffness_matrix



def loss_function_force(predicted_stress, experimental_force, strain, volume, displacement):
    # Custom loss function based on the global balance of energy (equation 6 in paper)
    loss = torch.sum(predicted_stress * strain * volume) - experimental_force * displacement
    return loss.abs()

def loss_function_divergence(stress, grid_spacing):
    # Compute the divergence of the stress field using central differences numerical method (equation 9-11 in paper)
    div_stress_x = (stress[:, 1:] - stress[:, :-1]) / grid_spacing
    div_stress_y = (stress[1:, :] - stress[:-1, :]) / grid_spacing
    return torch.sum(div_stress_x ** 2 + div_stress_y ** 2)

def loss_function_boundary(stress, boundary_mask):
    # Enforce boundary conditions (e.g., zero traction at free boundaries)
    return torch.sum((stress * boundary_mask) ** 2)

def pseudogradient_es_adam_optimization(model, train_data, num_iterations=500, population_size=10,
                                        mutation_rate=0.03, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    # Initialize the Adam parameters
    m = {key: torch.zeros_like(val) for key, val in model.state_dict().items()}
    v = {key: torch.zeros_like(val) for key, val in model.state_dict().items()}

    for iteration in range(num_iterations):
        population = [model.state_dict() for _ in range(population_size)]
        losses = []

        # Evaluate the loss for each candidate solution in the population
        for individual in population:
            model.load_state_dict(individual)
            loss = compute_total_loss(model, train_data)
            losses.append(loss.item())

        # Identify the best and worst performing solutions
        best_individual = population[np.argmin(losses)]
        worst_individual = population[np.argmax(losses)]

        # Calculate the pseudogradient
        pseudogradient = {}
        for key in best_individual.keys():
            pseudogradient[key] = (best_individual[key] - worst_individual[key])

        # Apply Adam optimization to the pseudogradient
        for key in pseudogradient.keys():
            m[key] = beta1 * m[key] + (1 - beta1) * pseudogradient[key]
            v[key] = beta2 * v[key] + (1 - beta2) * (pseudogradient[key] ** 2)

            m_hat = m[key] / (1 - beta1 ** (iteration + 1))
            v_hat = v[key] / (1 - beta2 ** (iteration + 1))

            update = learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)

            # Apply the update to the best individual's state in the population
            best_individual[key] -= update * mutation_rate

        # Update the model parameters with the best individual after Adam optimization
        model.load_state_dict(best_individual)

        # Perform a regular gradient descent step with the Adam-optimized pseudogradient
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer.zero_grad()
        total_loss = compute_total_loss(model, train_data, w_div=10, w_bc=10)
        total_loss.backward()
        optimizer.step()

        print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {total_loss.item()}")

def compute_total_loss(model, train_data, w_div, w_bc):
    """
    Args:
    - w_div: weight for divergence loss
    - w_bc: weight for boundary condition loss
    """
    predicted_stress = model(train_data['strain'])
    loss_force = loss_function_force(predicted_stress, train_data['force'], train_data['strain'], train_data['volume'],
                                     train_data['displacement'])
    loss_divergence = loss_function_divergence(predicted_stress, train_data['grid_spacing'])
    loss_boundary = loss_function_boundary(predicted_stress, train_data['boundary_mask'])

    total_loss = loss_force + w_div * loss_divergence + w_bc * loss_boundary
    return total_loss

def rotation_backpropagation(model, train_data, num_rotations=100):
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for rotation in range(num_rotations):
        # Randomly generate rotation matrices
        Q = generate_random_rotation_matrix()

        # Rotate the predicted strain using Q * strain * Q^T
        strain = train_data['strain']
        stress = model(strain)
        rotated_strain = apply_rotation(Q, strain)

        # Pass the rotated strain through the model to get the predicted stress
        predicted_stress = model(rotated_strain)

        # Rotate the predicted stress as well (optional, depending on the approach)
        rotated_stress = apply_rotation(Q, stress)

        # Train on augmented data (matching predicted rotated stress with the rotated stress tensor)
        optimizer.zero_grad()
        loss = torch.sum((predicted_stress - rotated_stress) ** 2)
        loss.backward()
        optimizer.step()

        print(f"Rotation {rotation + 1}/{num_rotations}, Loss: {loss.item()}")

def generate_random_rotation_matrix():
    theta = np.random.uniform(0, 2*np.pi)
    c, s = np.cos(theta), np.sin(theta)
    Q = np.array([[c, -s], [s, c]])
    return torch.tensor(Q, dtype=torch.float32)

def apply_rotation(Q, tensor):
    #Applies Q * tensor * Q^T
    return torch.matmul(Q, torch.matmul(tensor, Q.T))



# Assuming train_data is a dictionary containing the necessary inputs
# Ensure that train_data and test_data are defined with the correct structure
model = MaterialModelNN(input_size=6, hidden_size=32, output_size=6)
train_data = {
    'strain': torch.randn(100, 6),
    'force': torch.randn(100),
    'volume': torch.randn(100),
    'displacement': torch.randn(100),
    'grid_spacing': 0.01,
    'boundary_mask': torch.ones(100, 6),
    'stress': torch.randn(100, 6)  # For rotation_backpropagation
}

pseudogradient_es_adam_optimization(model, train_data, num_iterations=500)
rotation_backpropagation(model, train_data, num_rotations=100)

# Assuming test_data is defined similarly to train_data
test_loss = compute_total_loss(model, test_data)
print(f"Test Loss: {test_loss.item()}")

# Extract the effective stiffness matrix
effective_stiffness_matrix = model.get_effective_stiffness_matrix()
print("Effective Stiffness Matrix:")
print(effective_stiffness_matrix)