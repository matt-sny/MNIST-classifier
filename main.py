import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
from nn_classes import SimpleCNN
from data_visualizer import plot_loss_accuracy, plot_confusion_matrix, display_images
from nn_trainer import train_model, test_model
import os

# Set random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # mean & std for MNIST
])

# Create datasets and DataLoader objects
train_dataset_full = MNIST(root='./data', train=True, transform=transform, download=True)
train_dataset, val_dataset = random_split(train_dataset_full, [50000, 10000])
test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)

batch_size = 64

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Define model, loss function and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model & validate
n_epochs = 10
train_accuracies, train_losses, val_accuracies, val_losses = train_model(model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, n_epochs=n_epochs)
if os.path.exists("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth"))
else:
    raise FileNotFoundError("No pre-trained model found.")
test_accuracy, test_loss = test_model(model, test_loader=test_loader, criterion=criterion)

# Predict n_pred values
n_pred = len(test_dataset)
model.eval()
with torch.no_grad():
    sample_images = [test_dataset[i][0] for i in range(n_pred)]
    true_labels = [test_dataset[i][1] for i in range(n_pred)]
    sample_images = torch.stack(sample_images)
    outputs = model(sample_images)
    _, predictions = torch.max(outputs, 1)
predictions = predictions.tolist()

# Get wrong predictions
wrong_predictions = [pred != true for pred, true in zip(predictions, true_labels)]
wrong_indices = [i for i, is_wrong in enumerate(wrong_predictions) if is_wrong]
wrong_guesses = [predictions[i] for i in wrong_indices]
print(f"Total samples: {len(predictions)} | Misclassified: {len(wrong_indices)}")

# Visualize results
plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies)
plot_confusion_matrix(model, test_loader)
display_images([test_dataset[i] for i in wrong_indices[:20]], predictions=wrong_guesses[:20])