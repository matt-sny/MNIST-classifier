import torch
import time


def train_model(model, train_loader, criterion, optimizer, n_epochs: int = 10, val_loader=None):
    """
    Multi-class classifier trainer.
    Expects model outputs of shape [N, C] and integer class labels in [0, C-1].
    """
    start_time = time.time()
    train_accuracy_list = []
    train_loss_list = []
    val_accuracy_list = []
    val_loss_list = []

    for epoch in range(n_epochs):
        model.train()
        train_correct: int = 0
        total_train_loss: float = 0.0
        total_train_samples: int = 0

        for images, labels in train_loader:

            labels_mc = labels.long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_mc)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Metrics
            preds = outputs.argmax(dim=1)
            correct = (preds == labels_mc).sum().item()
            batch_size = labels_mc.size(0)

            total_train_loss += loss.item() * batch_size
            train_correct += correct
            total_train_samples += batch_size

        avg_train_loss = total_train_loss / max(total_train_samples, 1)
        train_loss_list.append(avg_train_loss)
        train_accuracy = train_correct / max(total_train_samples, 1)
        train_accuracy_list.append(train_accuracy)

        # Validation loop
        if val_loader is not None:
            model.eval()
            val_correct: int = 0
            total_val_loss: float = 0.0
            total_val_samples: int = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    labels_mc = labels.long()
                    outputs = model(images)
                    loss = criterion(outputs, labels_mc)

                    preds = outputs.argmax(dim=1)
                    correct = (preds == labels_mc).sum().item()
                    batch_size = labels_mc.size(0)

                    total_val_loss += loss.item() * batch_size
                    val_correct += correct
                    total_val_samples += batch_size

            avg_val_loss = total_val_loss / max(total_val_samples, 1)
            val_loss_list.append(avg_val_loss)
            val_accuracy = val_correct / max(total_val_samples, 1)
            val_accuracy_list.append(val_accuracy)

            print(
                f"Epoch [{epoch+1}/{n_epochs}], Train loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
            )
        else:
            print(
                f"Epoch [{epoch+1}/{n_epochs}], Train loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}"
            )

    if val_loader is None:
        val_accuracy_list = None
        val_loss_list = None

    total_time = time.time() - start_time
    print(f"Training complete. Total training time: {total_time//60:.0f} minutes {total_time%60:.1f} seconds")
    return train_accuracy_list, train_loss_list, val_accuracy_list, val_loss_list