import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from layers import FullyConnectedLayer, LogLoss, LogisticSigmoidLayer, InputLayer

# Load dataset
df = pd.read_excel('df_credit.xls')
df = df.drop(columns=['Unnamed: 0'])

# Convert all columns to numerical format
df = df.apply(pd.to_numeric, errors='coerce')

# Replace NaN values with the mean of the column
df = df.fillna(df.mean())

# Shuffle the rows to randomize the data
df = df.sample(frac=1).reset_index(drop=True)

# Extract input matrix X and target variable y
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)

# Standardize X using mean and std from the training data
mean = np.mean(X, axis=0)
std = np.std(X, axis=0, ddof=1)  # Use ddof=1 for sample standard deviation
X = (X - mean) / (std + 1e-8)

# Manually split the data: 2/3 for training and 1/3 for validation
n = len(X)
train_size = int(2 / 3 * n)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Initialize layers
input_layer = InputLayer(X)
fullyConnected_layer = FullyConnectedLayer(X_train.shape[1], 1)
sigmoid_layer = LogisticSigmoidLayer()
log_loss = LogLoss()

# Combine layers in a list for easier management
layers = [input_layer, fullyConnected_layer, sigmoid_layer]

# Set learning rate and termination criterion
learning_rate = 1e-3
prev_loss = np.inf
tolerance = 1e-10  # Set tolerance to 1e-10
max_epochs = 100000  # Set max_epochs to 100000

# To store loss values
train_losses = []
epoch = None
loss = None

# Training loop
for epoch in range(max_epochs):
    # Forward propagation
    h = X_train
    for layer in layers[1:]:
        h = layer.forward(h)

    # Compute loss
    loss = log_loss.eval(y_train, h)
    train_losses.append(loss)

    # Terminate if change in loss is below the tolerance
    if abs(prev_loss - loss) < tolerance:
        break

    # Backward propagation
    grad = log_loss.gradient(y_train, h)
    for i in range(len(layers) - 1, 0, -1):
        grad = layers[i].backward(grad)
        if isinstance(layers[i], FullyConnectedLayer):  # Update weights only for FullyConnectedLayer
            layers[i].updateWeights(grad, learning_rate)

    # Update previous loss
    prev_loss = loss

# Calculate training accuracy
y_train_pred = sigmoid_layer.forward(fullyConnected_layer.forward(X_train))
train_accuracy = np.mean((y_train_pred > 0.5) == y_train)

# Display results
print(f"Number of epochs: {epoch}")
print(f"Final training log loss: {loss}")
print(f"Training accuracy: {train_accuracy * 100:.2f}%")

# Plot training loss
plt.figure(figsize=(12, 6))
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss vs. Epoch')
plt.show()
