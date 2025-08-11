from pyexpat import model
import matplotlib.pyplot as plt
from math import sqrt, ceil
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def display_images(images, label_categories=None):
    """
    Display a single image or a grid of images.
    Each image can be a tensor, numpy array, or (image, label) tuple.
    If label_categories is provided, display the vegetable name for each label.
    """

    # Accept a single image or a list of images
    if not isinstance(images, (list, tuple)) or (isinstance(images, tuple) and hasattr(images[0], 'shape')):
        images = [images]

    n_images = len(images)
    cols = 1 if n_images == 1 else ceil(sqrt(n_images))
    rows = 1 if n_images == 1 else int((n_images + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 3))
    axes = np.array(axes).reshape(-1) if n_images > 1 else [axes]

    for i, img_item in enumerate(images):
        if isinstance(img_item, tuple):
            image, label = img_item
        else:
            image = img_item
            label = None

        # Convert torch.Tensor to numpy for display
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu()
            # Unnormalize if needed
            if image.ndim == 3 and image.shape[0] == 3:
                # Default ImageNet mean/std
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image = image * std + mean
            image = image.numpy()
            # If image is CHW, convert to HWC
            if image.ndim == 3 and image.shape[0] in (1, 3):
                image = np.transpose(image, (1, 2, 0))
            # If image is single-channel, squeeze last dim
            if image.shape[-1] == 1:
                image = image.squeeze(-1)

        axes[i].imshow(image)
        axes[i].axis('off')

        display_label = label
        if label is not None and label_categories is not None:
            get_key_from_value = lambda d, value: next((k for k, v in d.items() if v == value), None)
            name = get_key_from_value(label_categories, label)
            if name is not None:
                display_label = name
        if display_label is not None:
            axes[i].set_title(str(display_label), fontsize=12)

    # Hide unused axes
    for j in range(n_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel('Epochs')
    ax1.grid(True, which='both')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(epochs, train_losses, label='Train Loss', color='tab:blue', marker='o')
    ax1.plot(epochs, val_losses, label='Validation Loss', color='tab:cyan', marker='o')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.grid(True, which='both')
    ax2.set_ylabel('Accuracy', color='tab:orange')
    ax2.plot(epochs, train_accuracies, label='Train Accuracy', color='tab:orange', marker='o')
    ax2.plot(epochs, val_accuracies, label='Validation Accuracy', color='tab:pink', marker='o')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.tight_layout()
    fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.85))
    plt.title('Training and Validation Loss/Accuracy')
    plt.show()
    
def plot_confusion_matrix(model, test_loader):
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(10)])
    fig, ax = plt.subplots(figsize=(8,8))
    disp.plot(ax=ax, cmap='Blues')
    plt.title("MNIST Test Set Confusion Matrix")
    plt.show()