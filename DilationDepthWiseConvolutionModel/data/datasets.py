from torch.utils.data import Dataset
import numpy as np

class BaseDataset(Dataset):
    def __init__(self, dataset, transform=None):
        """
        Base dataset class for applying transformations to datasets.

        Args:
            dataset: The dataset to use (e.g., CIFAR10, ImageFolder).
            transform: Transformation to apply to images.
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Gets an item from the dataset and applies the transformation.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            Tuple: (transformed_image, label)
        """
        image, label = self.dataset[idx]
        image = np.array(image)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label
