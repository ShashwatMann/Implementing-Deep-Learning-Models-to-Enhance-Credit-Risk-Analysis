import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from layers1 import FullyConnectedLayer, LogLoss, LogisticSigmoidLayer, ReLuLayer

# Load dataset
df = pd.read_excel('df_credit.xls')
df = df.drop(columns=['Unnamed: 0'])
df = df.apply(pd.to_numeric, errors='coerce')
df = df.fillna(df.mean())
df = df.sample(frac=1).reset_index(drop=True)

X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)

mean = np.mean(X, axis=0)
std = np.std(X, axis=0, ddof=1)
X = (X - mean) / (std + 1e-8)

n = len(X)
train_size = int(1 / 5 * n)  # Reduced from 2/3 of n to 1/5 of n
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Define Network Architecture
hidden_neurons1 = 128
hidden_neurons2 = 256
hidden_neurons3 = 512
fullyConnected_layer1 = FullyConnectedLayer(X_train.shape[1], hidden_neurons1, initialization='xavier',
                                            optimizer='adam')
ReLu_Layer1 = ReLuLayer()
fullyConnected_layer2 = FullyConnectedLayer(hidden_neurons1, hidden_neurons2, initialization='xavier', optimizer='adam')
ReLu_Layer2 = ReLuLayer()
fullyConnected_layer3 = FullyConnectedLayer(hidden_neurons2, hidden_neurons3, initialization='xavier', optimizer='adam')
ReLu_Layer3 = ReLuLayer()
fullyConnected_layer4 = FullyConnectedLayer(hidden_neurons3, 1, initialization='xavier', optimizer='adam')
sigmoid_layer = LogisticSigmoidLayer()
log_loss = LogLoss()

network = [fullyConnected_layer1, ReLu_Layer1, fullyConnected_layer2, ReLu_Layer2, fullyConnected_layer3, ReLu_Layer3,
           fullyConnected_layer4, sigmoid_layer, log_loss]

epochs = 20000  # Fixed number of epochs
train_losses = []
val_losses = []
n_last_epochs = 10  # Number of last epochs to consider for early stopping
tolerance = 1e-6  # Tolerance level

epoch = None
loss = None
h_train = None
h_val = None

# Training loop
for epoch in range(epochs):
    # Forward propagation
    h_train = X_train
    for layer in network[:-1]:
        h_train = layer.forward(h_train)

    # Compute training loss
    loss = log_loss.eval(y_train, h_train)
    train_losses.append(loss)

    # Backward propagation
    grad = log_loss.gradient(y_train, h_train)
    for layer in reversed(network[1:-1]):
        grad = layer.backward(grad)

    # Update weights
    for layer in network[1:-1]:
        if hasattr(layer, 'update'):
            gradW = layer.train_input.T @ grad / len(layer.train_input)
            gradB = np.sum(grad, axis=0, keepdims=True)
            layer.updateWeightsWithAdam(gradW, gradB)

    # Validation
    h_val = X_val
    for layer in network[:-1]:
        h_val = layer.forward(h_val)

    val_loss = log_loss.eval(y_val, h_val)
    val_losses.append(val_loss)

    # Check for early stopping
    if epoch >= 2 * n_last_epochs - 1:
        prev_mean_loss = np.mean(train_losses[-2 * n_last_epochs:-n_last_epochs])
        curr_mean_loss = np.mean(train_losses[-n_last_epochs:])
        diff = abs(prev_mean_loss - curr_mean_loss)
        print(
            f"Epoch {epoch}: Previous mean loss: {prev_mean_loss}, Current mean loss: {curr_mean_loss}, Difference: {diff}")  # Debugging line


# Calculate training and validation accuracy
y_train_pred = (h_train > 0.5).astype(int)
y_val_pred = (h_val > 0.5).astype(int)

train_accuracy = np.mean(y_train_pred == y_train) * 100
val_accuracy = np.mean(y_val_pred == y_val) * 100

# Display results
print(f"Number of epochs: {epoch + 1}")
print(f"Final training log loss: {loss}")
print(f"Training accuracy: {train_accuracy:.2f}%")
print(f"Validation accuracy: {val_accuracy:.2f}%")

# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
