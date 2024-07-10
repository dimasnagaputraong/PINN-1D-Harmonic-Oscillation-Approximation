import itertools
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calculate_damping_quality_kpi(k, d, m):
    eigenfrequency = np.sqrt(k / m) / (2 * np.pi)
    damping_ratio = d / (2 * np.sqrt(k * m))
    quality_factor = 1 / (2 * damping_ratio)
    return eigenfrequency, damping_ratio, quality_factor

def generate_and_evaluate_combinations(c_F_values, d_F_values, m):
    combinations = list(itertools.product(c_F_values, d_F_values))
    results = []
    for c_F, d_F in combinations:
        k = c_F  # Example stiffness from c_F (you may need to adjust based on your model)
        d = d_F  # Example damping from d_F (you may need to adjust based on your model)
        eigenfrequency, damping_ratio, quality_factor = calculate_damping_quality_kpi(k, d, m)
        results.append((c_F, d_F, eigenfrequency, damping_ratio, quality_factor))
    return results

# Define realistic parameter ranges
c_F_values = np.linspace(10000, 200000, 10)  # Example values for chassis stiffness
d_F_values = np.linspace(500, 5000, 10)   # Example values for chassis damping


# Example mass (you can adjust this)
mass = 537  # kg, example mass for a quarter-car body

# Generate and evaluate combinations
results = generate_and_evaluate_combinations(c_F_values, d_F_values, mass)

# Extract KPI values from results for clustering
kpi_values = np.array([[eigenfrequency, damping_ratio, quality_factor] for _, _, eigenfrequency, damping_ratio, quality_factor in results])

# Perform k-means clustering
n_clusters = 20  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(kpi_values)

# Select the point closest to the centroid from each cluster
selected_combinations = []
for i in range(n_clusters):
    cluster_indices = np.where(kmeans.labels_ == i)[0]
    cluster_points = kpi_values[cluster_indices]
    centroid = kmeans.cluster_centers_[i]
    closest_point_index = cluster_indices[np.argmin(np.linalg.norm(cluster_points - centroid, axis=1))]
    selected_combinations.append(results[closest_point_index])


# Print selected combinations and their KPIs
for combination in selected_combinations:
    print("Parameters:", combination[:2], "KPIs:", combination[2:])

# Visualize the clusters
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of all KPI values
sc = ax.scatter(kpi_values[:, 0], kpi_values[:, 1], kpi_values[:, 2], c=kmeans.labels_, cmap='viridis', label='Cluster Points', alpha=0.6)

# Highlight the selected representative combinations
selected_kpi_values = np.array([combination[2:] for combination in selected_combinations])
ax.scatter(selected_kpi_values[:, 0], selected_kpi_values[:, 1], selected_kpi_values[:, 2], c='red', s=150, edgecolor='black', marker='o', label='Selected Points')

ax.set_xlabel('Eigenfrequenz (Hz)')
ax.set_ylabel('Dämpfungsgrad')
ax.set_zlabel('Gütefaktor')
plt.title('KPI Clusters and Selected Representative Combinations')
plt.legend()
plt.colorbar(sc, label='Cluster Label')
plt.show()
