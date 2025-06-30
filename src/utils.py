import torch
import matplotlib.pyplot as plt
import numpy as np

def show_image(img_tensor, title=""):
    """
    Display an image tensor using matplotlib.

    Args:
        img_tensor (torch.Tensor): Image tensor in (C, H, W) format.
        title (str, optional): Title for the plot. Defaults to "".
    """
    # Rearrange dimensions from (C, H, W) to (H, W, C)
    img = img_tensor.permute(1, 2, 0).cpu().numpy()

    # Display the image
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()


def load_model(model, path, device):
    """
    Load model weights from a file and prepare the model for evaluation.

    Args:
        model (torch.nn.Module): The model architecture.
        path (str): Path to the saved model weights (.pth file).
        device (torch.device): Device to map the model to (CPU or CUDA).

    Returns:
        torch.nn.Module: The loaded and ready-to-use model.
    """
    # Load saved state dict
    model.load_state_dict(torch.load(path, map_location=device))

    # Move model to the specified device
    model.to(device)

    # Set to evaluation mode
    model.eval()

    return model
