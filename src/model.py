from torchvision.models import vit_b_16
import torch.nn as nn

def build_model(num_classes):
    """
    Builds a Vision Transformer (ViT) model for image classification.

    Args:
        num_classes (int): Number of output classes for the classification task.

    Returns:
        model (torch.nn.Module): The modified ViT model with a new classification head.
    """

    # Load the pre-trained ViT-B/16 model from torchvision (pretrained on ImageNet)
    model = vit_b_16(pretrained=True)

    # Replace the final classification head to match the desired number of classes
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    # Return the modified model
    return model
