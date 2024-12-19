from .data_loader import BaseDataLoader
from .cifar10_dataset import get_cifar10_datasets
import json

class CIFAR10DataLoader(BaseDataLoader):
    def __init__(self,
                 batch_size=64, num_workers=4,
                 config_path=".\config\config.json"):
        """
        Initializes the CIFAR-10 DataLoader.

        Args:
            train_transform_params (dict): Parameters for training data transformations.
            test_transform_params (dict): Parameters for test data transformations.
            batch_size (int): The batch size for loading data. Default is 64.
            num_workers (int): The number of worker threads for data loading. Default is 4.
        """
        # Load configuration parameters from config.json
        with open(config_path, 'r') as f:
            config = json.load(f)

        batch_size = config['batch_size']
        num_workers = config['num_workers']
        train_transform_params={}
        test_transform_params={}

        # Get CIFAR-10 datasets
        train_dataset, test_dataset = get_cifar10_datasets()
        
        # Initialize the BaseDataLoader class
        super().__init__(train_dataset, test_dataset, batch_size, num_workers)
