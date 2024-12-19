from albumentations import Compose, HorizontalFlip, ShiftScaleRotate, CoarseDropout, Normalize
from torchvision.datasets import CIFAR10
from .datasets import BaseDataset
from albumentations.pytorch import ToTensorV2

def get_cifar10_datasets():
    """
    Returns train and test datasets for the CIFAR-10 dataset.

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Define the transformations for training and testing datasets
    train_transform = Compose([
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        CoarseDropout(
            max_holes=1, max_height=16, max_width=16, 
            min_holes=1, min_height=16, min_width=16, 
            fill_value=mean, mask_fill_value=None, p=0.5
        ),
        Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    test_transform = Compose([
        Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    # Load the CIFAR-10 dataset
    train_dataset = CIFAR10(root="./data", train=True, download=True)
    test_dataset = CIFAR10(root="./data", train=False, download=True)

    # Use the BaseDataset class for both train and test datasets
    return (
        BaseDataset(train_dataset, transform=train_transform),
        BaseDataset(test_dataset, transform=test_transform)
    )
