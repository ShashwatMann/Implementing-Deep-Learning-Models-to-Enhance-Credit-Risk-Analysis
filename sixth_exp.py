import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras

Sequential = keras.models.Sequential
LSTM = keras.layers.LSTM
GRU = keras.layers.GRU
Dense = keras.layers.Dense

# Load and preprocess the dataset
df = pd.read_excel('df_credit.xls')
df = df.drop(columns=['Unnamed: 0'])
df = df.apply(pd.to_numeric, errors='coerce')
df = df.fillna(df.mean())
df = df.sample(frac=1).reset_index(drop=True)

X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

mean = np.mean(X, axis=0)
std = np.std(X, axis=0, ddof=1)
X = (X - mean) / (std + 1e-8)

# Reshape X to have 3 dimensions for LSTM/GRU (samples, timesteps, features)
X = X.reshape(X.shape[0], 1, X.shape[1])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# LSTM Model
lstm_model = Sequential([
    LSTM(32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_history = lstm_model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_val, y_val))

# GRU Model
gru_model = Sequential([
    GRU(32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    GRU(64),
    Dense(1, activation='sigmoid')
])

gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gru_history = gru_model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_val, y_val))

# Plotting Losses
plt.figure(figsize=(12, 6))

# LSTM loss plot
plt.plot(lstm_history.history['loss'], label='LSTM Training Loss')
plt.plot(lstm_history.history['val_loss'], label='LSTM Validation Loss')

# GRU loss plot
plt.plot(gru_history.history['loss'], label='GRU Training Loss')
plt.plot(gru_history.history['val_loss'], label='GRU Validation Loss')

plt.title('Training and Validation Loss vs. Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Plotting Accuracies
plt.figure(figsize=(12, 6))

# LSTM accuracy plot
plt.plot(lstm_history.history['accuracy'], label='LSTM Training Accuracy')
plt.plot(lstm_history.history['val_accuracy'], label='LSTM Validation Accuracy')

# GRU accuracy plot
plt.plot(gru_history.history['accuracy'], label='GRU Training Accuracy')
plt.plot(gru_history.history['val_accuracy'], label='GRU Validation Accuracy')

plt.title('Training and Validation Accuracy vs. Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Extracting details for LSTM
lstm_avg_train_accuracy = np.mean(lstm_history.history['accuracy']) * 100
lstm_avg_val_accuracy = np.mean(lstm_history.history['val_accuracy']) * 100
lstm_epochs_count = len(lstm_history.history['accuracy'])

# Extracting details for GRU
gru_avg_train_accuracy = np.mean(gru_history.history['accuracy']) * 100
gru_avg_val_accuracy = np.mean(gru_history.history['val_accuracy']) * 100
gru_epochs_count = len(gru_history.history['accuracy'])

# Displaying the results
print(f"LSTM Model Results:")
print(f"Average Training Accuracy: {lstm_avg_train_accuracy:.2f}%")
print(f"Average Validation Accuracy: {lstm_avg_val_accuracy:.2f}%")
print(f"Total Epochs: {lstm_epochs_count}\n")

print(f"GRU Model Results:")
print(f"Average Training Accuracy: {gru_avg_train_accuracy:.2f}%")
print(f"Average Validation Accuracy: {gru_avg_val_accuracy:.2f}%")
print(f"Total Epochs: {gru_epochs_count}\n")
