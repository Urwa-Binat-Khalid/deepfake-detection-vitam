import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def get_dataloaders(data_dir, batch_size=32):

    # Define transformations for training data: resize, randomly flip, and convert to tensor
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.ToTensor(),
    ])

    # Define transformations for test data: only resize and convert to tensor
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load training and test datasets using ImageFolder
    train_data = ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    test_data = ImageFolder(os.path.join(data_dir, 'test'), transform=test_transforms)

    # Create DataLoaders for batching and shuffling
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Return DataLoaders and class names
    return train_loader, test_loader, train_data.classes
