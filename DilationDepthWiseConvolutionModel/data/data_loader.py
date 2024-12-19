from torch.utils.data import DataLoader

class BaseDataLoader:
    def __init__(self, train_dataset, test_dataset, batch_size=64, num_workers=4):
        """
        Initializes the Base DataLoader class.

        Args:
            train_dataset (Dataset): The training dataset.
            test_dataset (Dataset): The testing dataset.
            batch_size (int): The batch size for loading data. Default is 64.
            num_workers (int): The number of worker threads for data loading. Default is 4.
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_train_loader(self):
        """
        Returns the DataLoader for the training dataset.

        Returns:
            DataLoader: The DataLoader for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def get_test_loader(self):
        """
        Returns the DataLoader for the test dataset.

        Returns:
            DataLoader: The DataLoader for the test dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
