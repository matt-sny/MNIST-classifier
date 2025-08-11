import matplotlib.pyplot as plt
from math import sqrt, ceil
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections.abc import Iterable

def display_images(images, *, mean_std=(0.1307, 0.3081), cmap='gray'):
    """Display MNIST (or generic) images.

    Accepts:
      - A single tensor / numpy array (C,H,W) or (H,W) or (H,W,1)
      - A batch tensor (N,C,H,W)
      - A list/tuple of tensors / numpy arrays
      - A list/tuple of (image, label) pairs

    Parameters
    ----------
    images : various
        See above.
    mean_std : (float, float) | None
        Mean & std used for normalization (default MNIST). If None, no un-normalization.
    n_max : int | None
        If provided and images is a batch/list, limit how many are shown.
    cmap : str
        Matplotlib colormap to use for single-channel images.
    """

    # If a batch tensor (N,C,H,W) provided, split into list
    if isinstance(images, torch.Tensor) and images.ndim == 4:
        images = list(images)

    # Expand generic iterable/generator
    if not isinstance(images, (list, tuple)):
        if isinstance(images, torch.Tensor):
            images = [images]
        elif isinstance(images, Iterable) and not isinstance(images, (str, bytes)):
            images = list(images)
        else:
            images = [images]

    n_images = len(images)
    cols = 1 if n_images == 1 else ceil(sqrt(n_images))
    rows = 1 if n_images == 1 else int((n_images + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.array(axes).reshape(-1) if n_images > 1 else [axes]

    mnist_mean, mnist_std = (mean_std if mean_std is not None else (None, None))

    for i, img_item in enumerate(images):
        if isinstance(img_item, tuple):
            image, label = img_item
        else:
            image, label = img_item, None

        # Tensor -> numpy
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu()
            # Unnormalize single-channel if stats provided and shape matches
            if mean_std is not None:
                if image.ndim == 3 and image.shape[0] == 1:  # (1,H,W)
                    image = image * mnist_std + mnist_mean
                elif image.ndim == 2:  # (H,W)
                    image = image * mnist_std + mnist_mean
            image = image.numpy()

        # Proceed only if numpy array
        if isinstance(image, np.ndarray):
            # Handle channel ordering
            if image.ndim == 3 and image.shape[0] in (1, 3):  # CHW -> HWC
                image = np.transpose(image, (1, 2, 0))
            if image.ndim == 3 and image.shape[2] == 1:  # HWC with 1 channel
                image = image.squeeze(2)

            # Clip for safety (after un-normalization)
            if image.ndim == 2:
                disp_img = np.clip(image, 0, 1)
                axes[i].imshow(disp_img, cmap=cmap)
            else:
                disp_img = np.clip(image, 0, 1)
                axes[i].imshow(disp_img)
        else:  # Fallback: show nothing / placeholder
            axes[i].text(0.5, 0.5, 'N/A', ha='center', va='center')
        axes[i].axis('off')
        if label is not None:
            axes[i].set_title(str(label), fontsize=10)

    # Hide unused axes
    for j in range(n_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel('Epochs')
    #ax1.grid(True, which='both')
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