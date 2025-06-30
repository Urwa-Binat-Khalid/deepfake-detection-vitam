import torch
from tqdm import tqdm
from src.evaluate import evaluate  # Assumes you have a separate evaluate() function

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device):
    """
    Trains the model for one epoch using mixed precision.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (loss): Loss function.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision.
        device (torch.device): Device to run on (CPU or CUDA).

    Returns:
        float: Average training loss for the epoch.
    """

    model.train()  # Set model to training mode
    total_loss = 0  # Track total loss for averaging

    for inputs, labels in tqdm(dataloader):
        # Move inputs and labels to the target device
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Reset gradients

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # Backward pass and optimizer step with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)  # Return average loss


def train_model(model, train_loader, test_loader, optimizer, criterion, scaler, device,
                epochs=10, patience=2, save_path='best_vit_model.pth'):
    """
    Full training loop with evaluation and early stopping.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Validation/test data loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (loss): Loss function.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler.
        device (torch.device): Device to run on.
        epochs (int): Number of epochs.
        patience (int): Early stopping patience.
        save_path (str): File path to save the best model.

    Returns:
        None
    """

    best_acc = 0  # Best validation accuracy seen so far
    early_stop_counter = 0  # Counts how many epochs since last improvement

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)

        # Evaluate on validation/test set
        test_acc = evaluate(model, test_loader, device)

        print(f"Train Loss: {train_loss:.4f} | Val Accuracy: {test_acc:.4f}")

        # Check if validation accuracy improved
        if test_acc > best_acc:
            best_acc = test_acc
            early_stop_counter = 0  # Reset counter
            torch.save(model.state_dict(), save_path)  # Save best model
            print("Model saved.")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break
