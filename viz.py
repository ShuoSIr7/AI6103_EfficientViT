import pandas as pd

# Load the data from the .logs file
log_data = pd.read_csv('training.logs')

# Extract metrics
epochs = log_data['Epoch'].values
train_losses = log_data['Train Loss'].values
train_accuracies = log_data['Train Acc'].values
val_losses = log_data['Val Loss'].values
val_accuracies = log_data['Val Acc'].values


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Plot training and validation loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss(SoftTarget)', color='blue')
plt.plot(epochs, val_losses, label='Val Loss(CrossEntropy)', color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train Acc@1', color='blue')
plt.plot(epochs, val_accuracies, label='Val Acc@1', color='orange')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
