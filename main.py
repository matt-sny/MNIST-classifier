import os
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet18_Weights
import numpy as np
import matplotlib.pyplot as plt
from data_loader import create_dataloaders, MyDataset
from data_visualizer import display_images, plot_loss_accuracy
from nn_trainer import train_model
import kagglehub
from typing import cast

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

data_path = os.path.join(kagglehub.dataset_download("misrakahmed/vegetable-image-dataset"), "Vegetable Images")
print(data_path)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

label_categories = {'Bean': 0, 'Bitter_Gourd': 1, 'Bottle_Gourd': 2, 'Brinjal': 3, 'Broccoli': 4, 
                        'Cabbage': 5, 'Capsicum': 6,  'Carrot': 7, 'Cauliflower': 8, 'Cucumber': 9, 
                        'Papaya': 10, 'Potato': 11, 'Pumpkin': 12, 'Radish': 13, 'Tomato': 14}
train_size = 1500
val_size = 200
test_size = 200
batch_size = 32
split_sizes = (train_size, val_size, test_size)

train_loader, val_loader, test_loader = create_dataloaders(data_path=data_path,
                                              label_categories=label_categories,
                                              split_sizes=split_sizes,
                                              train_transform=train_transform, 
                                              val_transform=val_transform,
                                              test_transform=val_transform,
                                              batch_size=batch_size)

# Create model
model1 = models.resnet18(weights=ResNet18_Weights.DEFAULT)

for param in model1.parameters():
    param.requires_grad = False

model1.fc = nn.Linear(model1.fc.in_features, 15)
for param in model1.fc.parameters():
    param.requires_grad = True

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model1.fc.parameters(), lr=0.001, weight_decay=1e-4)

# Train model & validate
n_epochs = 5
train_accuracies, train_losses, val_accuracies, val_losses = train_model(model1, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, n_epochs=n_epochs)

train_dataset = cast(MyDataset, train_loader.dataset)
val_dataset = cast(MyDataset, val_loader.dataset)
test_dataset = cast(MyDataset, test_loader.dataset)

# Visualize results
print(f"Train dataset size: {len(train_dataset)} ({len(train_loader)} batches each of up to {batch_size} images each)")
print(f"Validation dataset size: {len(val_dataset)} ({len(val_loader)} batches each of up to {batch_size} images each)")
print(f"Test dataset size: {len(test_dataset)} ({len(test_loader)} batches each of up to {batch_size} images each)")
#print(f"{train_loader.dataset[0]}")
plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies)
#display_images(val_dataset[139], label_categories=label_categories)