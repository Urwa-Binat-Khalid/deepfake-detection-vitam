import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate(model, dataloader, device):
    """
    Evaluate the model on a given DataLoader and return accuracy.

    Args:
        model (torch.nn.Module): Trained model to evaluate.
        dataloader (DataLoader): DataLoader for test/validation data.
        device (torch.device): Device to run on.

    Returns:
        float: Accuracy of the model on the dataset.
    """
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Get predicted class labels
            _, preds = torch.max(outputs, 1)

            # Count correct predictions
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total  # Compute accuracy


def evaluate_and_report(model, dataloader, class_names, device):
    """
    Evaluate the model and generate a classification report & confusion matrix.

    Args:
        model (torch.nn.Module): Trained model to evaluate.
        dataloader (DataLoader): DataLoader for test/validation data.
        class_names (list): List of class names for the report.
        device (torch.device): Device to run on.

    Saves:
        confusion_matrix.png: Confusion matrix heatmap in 'figures/' directory.
    """
    model.eval()  # Set model to evaluation mode
    y_true, y_pred = [], []

    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in dataloader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)

            # Get predicted class labels
            _, preds = torch.max(outputs, 1)

            # Collect true and predicted labels
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Print classification report
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot and save confusion matrix heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("figures/confusion_matrix.png")
    plt.close()
