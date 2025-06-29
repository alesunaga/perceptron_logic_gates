import codecademylib3_seaborn # Specific library for Codecademy's environment, often for styling plots
from sklearn.linear_model import Perceptron # Import the Perceptron model from scikit-learn
import matplotlib.pyplot as plt # Library for creating static, interactive, and animated visualizations
import numpy as np # Library for numerical operations, especially with arrays
from itertools import product # Used for creating Cartesian products, useful for grid generation

# --- Project Overview ---
# This script demonstrates how a Perceptron, a fundamental building block of neural networks,
# can be used to model basic logic gates: AND, OR, and XOR.
# It visualizes the input data, trains the Perceptron for each gate, evaluates its performance,
# and generates heatmaps to illustrate the decision boundary of the trained model.

# --- Modeling the AND Logic Gate ---

# Define the input data for the AND gate.
# The inputs are binary (0 or 1).
data = [[0, 0], [1, 0], [0, 1], [1, 1]]

# Define the expected outputs (labels) for the AND gate corresponding to the 'data' inputs.
# AND gate output is 1 only if both inputs are 1; otherwise, it's 0.
labels = [0, 0, 0, 1]

# Visualize the AND gate's input data points with their respective labels.
# This helps to understand the separability of the data.
x = [point[0] for point in data] # Extract x-coordinates from data
y = [point[1] for point in data] # Extract y-coordinates from data
plt.scatter(x, y, c=labels) # Plot points, colored by their labels
plt.title('AND Gate Inputs and Labels') # Add a title for clarity
plt.xlabel('Input 1') # Label x-axis
plt.ylabel('Input 2') # Label y-axis
plt.show() # Display the plot

# Initialize and train the Perceptron model for the AND gate.
# 'max_iter' sets the maximum number of passes over the training data (epochs).
# 'random_state' ensures reproducibility of the training process.
classifier = Perceptron(max_iter=40, random_state=22)
classifier.fit(data, labels) # Train the perceptron with the AND gate data and labels

# Evaluate the trained Perceptron's accuracy on the AND gate data.
# A score of 1.0 indicates perfect classification.
print("Perceptron score for AND gate:", classifier.score(data, labels))

# --- Modeling the XOR Logic Gate ---

# Define the expected outputs (labels) for the XOR gate.
# XOR gate output is 1 if inputs are different; otherwise, it's 0.
labels_xor = [0, 1, 1, 0]

# Train the same Perceptron classifier for the XOR gate.
# Note: A single-layer Perceptron cannot perfectly classify XOR due to its non-linear separability.
classifier.fit(data, labels_xor)
print("Perceptron score for XOR gate:", classifier.score(data, labels_xor))

# Visualize the XOR gate's input data points with their respective labels.
plt.scatter(x, y, c=labels_xor)
plt.title('XOR Gate Inputs and Labels')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.show()

# --- Modeling the OR Logic Gate ---

# Define the expected outputs (labels) for the OR gate.
# OR gate output is 1 if at least one input is 1; otherwise, it's 0.
labels_or = [0, 1, 1, 1]

# Train the Perceptron classifier for the OR gate.
classifier.fit(data, labels_or)
print("Perceptron score for OR gate:", classifier.score(data, labels_or))

# Visualize the OR gate's input data points with their respective labels.
plt.scatter(x, y, c=labels_or)
plt.title('OR Gate Inputs and Labels')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.show()

# --- Analyzing the Perceptron's Decision Function ---

# Retrain the classifier for the AND gate to demonstrate its decision function.
classifier.fit(data, labels)

# Use the `decision_function` to get the signed distance of samples to the hyperplane.
# A positive value typically means it's classified as the positive class (1),
# and a negative value as the negative class (0). The magnitude indicates confidence.
data_decision = [[0, 0], [1, 1], [0.5, 0.5]]
decision = classifier.decision_function(data_decision)
print("Decision function output for sample points (AND gate):", decision)

# --- Heatmap Visualization of Decision Boundaries ---
# These heatmaps show how the Perceptron classifies points across the entire input space (0 to 1).
# The color intensity represents the absolute distance to the decision boundary.
# A lower absolute distance indicates points closer to the boundary, where classification is less certain.

# --- Heatmap for AND Gate Decision Function ---
x_values = np.linspace(0, 1, 100) # Generate 100 points between 0 and 1 for the x-axis
y_values = np.linspace(0, 1, 100) # Generate 100 points between 0 and 1 for the y-axis
# Create a grid of all possible (x, y) combinations from the generated values
point_grid = list(product(x_values, y_values))

# Calculate the decision function for each point on the grid
distances = classifier.decision_function(point_grid)
abs_distances = [abs(distance) for distance in distances] # Get absolute distances for heatmap visualization

# Reshape the distances into a 2D matrix matching the grid dimensions
distances_matrix = np.reshape(abs_distances, (100, 100))

# Create a heatmap using `pcolormesh`.
# The color represents the `distances_matrix` values across the `x_values` and `y_values` grid.
heatmap = plt.pcolormesh(x_values, y_values, distances_matrix, cmap='viridis') # 'viridis' is a good color map
plt.colorbar(heatmap) # Add a color bar to indicate distance values
plt.title('AND Gate Decision Boundary Heatmap')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.show()

# --- Heatmap for OR Gate Decision Function ---
# Retrain the classifier for the OR gate before generating its heatmap
classifier.fit(data, labels_or)

x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)
point_grid = list(product(x_values, y_values))
distances = classifier.decision_function(point_grid)
abs_distances = [abs(distance) for distance in distances]
distances_matrix = np.reshape(abs_distances, (100, 100))
heatmap = plt.pcolormesh(x_values, y_values, distances_matrix, cmap='viridis')
plt.colorbar(heatmap)
plt.title('OR Gate Decision Boundary Heatmap')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.show()

# --- Heatmap for XOR Gate Decision Function ---
# Retrain the classifier for the XOR gate before generating its heatmap
classifier.fit(data, labels_xor)

x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)
point_grid = list(product(x_values, y_values))
distances = classifier.decision_function(point_grid)
abs_distances = [abs(distance) for distance in distances]
distances_matrix = np.reshape(abs_distances, (100, 100))
heatmap = plt.pcolormesh(x_values, y_values, distances_matrix, cmap='viridis')
plt.colorbar(heatmap)
plt.title('XOR Gate Decision Boundary Heatmap')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.show()
