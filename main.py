import os
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
from nn_classes import SimpleCNN
from data_visualizer import display_images, plot_loss_accuracy
from nn_trainer import train_model, test_model

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # mean & std for MNIST
])

train_dataset_full = MNIST(root='./data', train=True, transform=transform, download=True)
train_dataset, val_dataset = random_split(train_dataset_full, [50000, 10000])
test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)

batch_size = 64

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Create model
model = SimpleCNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model & validate
n_epochs = 10
train_accuracies, train_losses, val_accuracies, val_losses = train_model(model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, n_epochs=n_epochs)
test_accuracy, test_loss = test_model(model, test_loader=test_loader, criterion=criterion)

# Visualize results
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")
plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies)
#display_images(val_dataset[139])