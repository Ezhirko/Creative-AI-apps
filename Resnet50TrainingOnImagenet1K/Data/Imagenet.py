import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import torchvision
import torch
import numpy as np
import json

class AlbumentationsTransform:
    def __init__(self, transform):
        if not isinstance(transform, A.Compose):
            raise ValueError("Transform must be an instance of albumentations.Compose")
        self.transform = transform

    def __call__(self, image):
        augmented = self.transform(image=np.array(image))
        return augmented["image"]
    
def get_imagenet1K_data():
    root_dir = '/mnt/ebs-volume/Dataset/ILSVRC/Data/CLS-LOC'

    training_folder_name = root_dir + '/train'
    val_folder_name = root_dir + '/val'
    

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    # Define the transformations for training and testing datasets
    train_transform =  A.Compose([
        A.PadIfNeeded(
            min_height=70,min_width=70,border_mode=0,
            value=mean,p=1.0),
        A.OneOf([
            A.RandomCrop(height=64,width=64,p=0.9),
            A.CenterCrop(height=64,width=64,z=0.1)
        ], p=1.0),
        A.HorizontalFlip(p=0.7),
        A.CoarseDropout(
                max_holes=1, max_height=16, max_width=16, 
                min_holes=1, min_height=16, min_width=16, 
                fill_value=mean, mask_fill_value=None, p=0.5
            ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Resize(height=64, width=64),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()    
    ])

    wrapped_train_transform = AlbumentationsTransform(transform=train_transform)
    wrapped_test_transform = AlbumentationsTransform(transform=test_transform)

    train_dataset = torchvision.datasets.ImageFolder(root=training_folder_name, transform=wrapped_train_transform)
    test_dataset = torchvision.datasets.ImageFolder(root=val_folder_name, transform=wrapped_test_transform)
    return train_dataset,test_dataset

class ImagenetDataLoader():
    def __init__(self,batch_size=64, num_workers=4,
                 config_path="/home/ubuntu/Session9/config/config.json"):
        """
        Initializes the CIFAR-10 DataLoader.

        Args:
            batch_size (int): The batch size for loading data. Default is 64.
            num_workers (int): The number of worker threads for data loading. Default is 4.
            cofiguration: The configuration file
        """
        # Load configuration parameters from config.json
        with open(config_path, 'r') as f:
            config = json.load(f)

        batch_size = config['batch_size']
        num_workers = config['num_workers']

        train_dataset, test_dataset = get_imagenet1K_data()
        self.train_loader = torch.utils.data.DataLoader(
                            train_dataset,
                            batch_size=batch_size,
                            num_workers = num_workers,
                            pin_memory=True,
                        )
        self.test_loader = torch.utils.data.DataLoader(
                            test_dataset,
                            batch_size=batch_size,
                            num_workers = num_workers,
                            pin_memory=True,
                        )


    def get_train_loader(self):
        return self.train_loader
    
    def get_test_loader(self):
        return self.test_loader
        