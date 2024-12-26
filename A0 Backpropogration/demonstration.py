import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Activation Function (Sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of Sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Forward propagation
def forward_propagation(X, weights):
    return sigmoid(np.dot(X, weights))

# Error Calculation (Mean Squared Error)
def calculate_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Backpropagation
def backpropagate(X, y_true, y_pred, weights, learning_rate):
    error = y_true - y_pred
    d_weights = np.dot(X.T, error * sigmoid_derivative(y_pred))
    weights += learning_rate * d_weights
    return weights, calculate_error(y_true, y_pred)

# Generate input data (simple XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input dataset
y = np.array([[0], [1], [1], [0]])  # Expected outputs (XOR)

# Initialize weights randomly
np.random.seed(42)
weights = np.random.rand(2, 1)

# Hyperparameters
learning_rate = 0.1
epochs = 100  # Number of iterations
errors = []
predictions = []

# Setup for the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

# Line objects for updating in animation
error_line, = ax1.plot([], [], 'b-', label='Error')
prediction_lines = [ax2.plot([], [], label=f"Prediction {i+1}")[0] for i in range(4)]

# Set limits and labels for error plot
ax1.set_xlim(0, epochs)
ax1.set_ylim(0, 1)
ax1.set_title("Error over Epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Error")
ax1.legend()

# Set limits and labels for prediction plot
ax2.set_xlim(0, epochs)
ax2.set_ylim(0, 1)
ax2.set_title("Predictions over Epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Prediction")
ax2.legend()

# Update function for the animation
def update(frame):
    global weights
    # Forward pass
    y_pred = forward_propagation(X, weights)
    
    # Backpropagate and update weights
    weights, error = backpropagate(X, y, y_pred, weights, learning_rate)
    
    # Store error and predictions
    errors.append(error)
    predictions.append(y_pred.flatten())
    
    # Update error line
    error_line.set_data(np.arange(len(errors)), errors)
    
    # Update prediction lines for each input
    for i, line in enumerate(prediction_lines):
        line.set_data(np.arange(len(predictions)), [pred[i] for pred in predictions])
    
    return error_line, *prediction_lines

# Animation
ani = FuncAnimation(fig, update, frames=range(epochs), blit=True, repeat=False)

# Show plot
plt.tight_layout()
plt.show()
