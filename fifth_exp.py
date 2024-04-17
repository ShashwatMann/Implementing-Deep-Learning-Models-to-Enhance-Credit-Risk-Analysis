import numpy as np
import pandas as pd
from layers3 import FullyConnectedLayer, LogLoss, LogisticSigmoidLayer, ReLuLayer, DropoutLayer, \
    SMOTEDataAugmentationLayer
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel('df_credit.xls')
df = df.drop(columns=['Unnamed: 0'])
df = df.apply(pd.to_numeric, errors='coerce')
df = df.fillna(df.mean())
df = df.sample(frac=1).reset_index(drop=True)

X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)


# Function to standardize features
def standardize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)
    return (X - mean) / (std + 1e-8)


# Function for data augmentation
def augmentData(X, augmentationLayer):
    augmentedSamples = []
    for i in range(X.shape[0]):
        neighbors = augmentationLayer.findKNeighbors(X, X[i])
        chosenNeighbor = neighbors[np.random.choice(augmentationLayer.kNeighbors)]
        alpha = np.random.random()
        syntheticSample = augmentationLayer.interpolate(X[i], chosenNeighbor, alpha)
        augmentedSamples.append(syntheticSample)
    return np.vstack(augmentedSamples)


# Function for plotting training and validation losses
def plot_losses(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss vs Epoch")
    plt.legend()
    plt.show()


# Initialize network
def initialize_network(dataIn):
    L1 = FullyConnectedLayer(dataIn.shape[1], hidden_neurons1, initialization='xavier', optimizer='adam')
    L1_ReLu = ReLuLayer()
    L1_Dropout = DropoutLayer(dropoutProb=dropout_rate)
    L2 = FullyConnectedLayer(hidden_neurons1, hidden_neurons2, initialization='xavier', optimizer='adam')
    L2_ReLu = ReLuLayer()
    L2_Dropout = DropoutLayer(dropoutProb=dropout_rate)
    L3 = FullyConnectedLayer(hidden_neurons2, hidden_neurons3, initialization='xavier', optimizer='adam')
    L3_ReLu = ReLuLayer()
    L3_Dropout = DropoutLayer(dropoutProb=dropout_rate)
    L4 = FullyConnectedLayer(hidden_neurons3, 1, initialization='xavier', optimizer='adam')
    L5 = LogisticSigmoidLayer()
    L6 = LogLoss()
    return [L1, L1_ReLu, L1_Dropout, L2, L2_ReLu, L2_Dropout, L3, L3_ReLu, L3_Dropout, L4, L5, L6]


# Network Hyperparameters
hidden_neurons1 = 128
hidden_neurons2 = 256
hidden_neurons3 = 512
dropout_rate = 0.2
epochs = 100  # Reduced epochs for demonstration

# 5-fold cross-validation
S = 5
folds = np.array_split(np.arange(len(X)), S)

all_train_accuracies = []
all_val_accuracies = []

for s in range(S):
    val_idx = folds[s]
    train_idx = np.concatenate([folds[i] for i in range(S) if i != s])

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    X_train = standardize_features(X_train)
    X_val = standardize_features(X_val)

    augLayer = SMOTEDataAugmentationLayer()
    X_train_augmented = augmentData(X_train, augLayer)

    network = initialize_network(X_train_augmented)

    train_losses = []
    val_losses = []
    h_train = None
    h_val = None
    log_loss_instance = LogLoss()

    for epoch in range(epochs):
        h_train = X_train_augmented
        for layer in network[:-1]:
            h_train = layer.forward(h_train)

        loss = log_loss_instance.eval(y_train, h_train)
        train_losses.append(loss)

        grad = log_loss_instance.gradient(y_train, h_train)
        for layer in reversed(network[1:-1]):
            grad = layer.backward(grad)

        for layer in network[1:-1]:
            if hasattr(layer, 'update'):
                gradW = layer.train_input.T @ grad / len(layer.train_input)
                gradB = np.sum(grad, axis=0, keepdims=True)
                layer.updateWeightsWithAdam(gradW, gradB)

        h_val = X_val
        for layer in network[:-1]:
            h_val = layer.forward(h_val)

        val_loss = log_loss_instance.eval(y_val, h_val)
        val_losses.append(val_loss)

    plot_losses(train_losses, val_losses)
    y_train_pred = (h_train > 0.5).astype(int)
    y_val_pred = (h_val > 0.5).astype(int)
    train_accuracy = np.mean(y_train_pred == y_train) * 100
    val_accuracy = np.mean(y_val_pred == y_val) * 100
    all_train_accuracies.append(train_accuracy)
    all_val_accuracies.append(val_accuracy)

    print(f"Fold-{s + 1} Training Accuracy: {train_accuracy:.2f}%")
    print(f"Fold-{s + 1} Validation Accuracy: {val_accuracy:.2f}%")
    print("-------------------------------")

average_train_accuracy = np.mean(all_train_accuracies)
average_val_accuracy = np.mean(all_val_accuracies)
print(f"Average Training Accuracy over {S}-Folds: {average_train_accuracy:.2f}%")
print(f"Average Validation Accuracy over {S}-Folds: {average_val_accuracy:.2f}%")
