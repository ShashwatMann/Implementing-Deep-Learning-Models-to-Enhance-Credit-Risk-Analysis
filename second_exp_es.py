import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from layers1 import FullyConnectedLayer, LogLoss, LogisticSigmoidLayer

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
train_size = int(2 / 3 * n)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

fullyConnected_layer = FullyConnectedLayer(X_train.shape[1], 1, initialization="xavier", optimizer="adam")
fullyConnected_layer.learning_rate = 1e-3
sigmoid_layer = LogisticSigmoidLayer()
log_loss = LogLoss()

epochs = 100000  # Fixed number of epochs
train_losses = []
val_losses = []
n_last_epochs = 10  # Number of last epochs to consider for early stopping
tolerance = 1e-6  # Tolerance level

epoch = None
loss = None
h_train = None
h_val = None

for epoch in range(epochs):
    # Forward propagation (training)
    h_train = fullyConnected_layer.forward(X_train)
    h_train = sigmoid_layer.forward(h_train)

    # Compute training loss
    loss = log_loss.eval(y_train, h_train)
    train_losses.append(loss)

    # Forward propagation (validation)
    h_val = fullyConnected_layer.forward(X_val)
    h_val = sigmoid_layer.forward(h_val)

    # Compute validation loss
    val_loss = log_loss.eval(y_val, h_val)
    val_losses.append(val_loss)

    # Backward propagation: remember to use h_train, not h_val
    grad = log_loss.gradient(y_train, h_train)
    sigmoid_layer.setPrevOut(h_train)
    grad = sigmoid_layer.backward(grad)

    # Update weights
    grad_w = X_train.T @ grad / len(X_train)
    grad_b = np.sum(grad, axis=0, keepdims=True)
    fullyConnected_layer.updateWeightsWithAdam(grad_w, grad_b)

    # Check for early stopping
    if epoch >= 2 * n_last_epochs - 1:
        prev_mean_loss = np.mean(train_losses[-2 * n_last_epochs:-n_last_epochs])
        curr_mean_loss = np.mean(train_losses[-n_last_epochs:])
        diff = abs(prev_mean_loss - curr_mean_loss)
        print(
            f"Epoch {epoch}: Previous mean loss: {prev_mean_loss}, Current mean loss: {curr_mean_loss}, Difference: {diff}")  # Debugging line
        if diff < tolerance:
            print("Early stopping criterion met.")
            break


# Calculate training and validation accuracy
y_train_pred = (h_train > 0.5).astype(int)
y_val_pred = (h_val > 0.5).astype(int)

train_accuracy = np.mean(y_train_pred == y_train) * 100
val_accuracy = np.mean(y_val_pred == y_val) * 100

# Display results
print(f"Number of epochs: {epoch+1}")
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
