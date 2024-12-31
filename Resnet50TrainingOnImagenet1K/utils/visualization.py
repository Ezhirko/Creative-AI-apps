import matplotlib.pyplot as plt
import torchvision
import torch

# Function to display a grid of images
def show_images_grid(loader, num_images=25, grid_size=(5, 5)):
    # Get a batch of images from the loader
    data_iter = iter(loader)
    images, labels = next(data_iter)
    
    # Select the first `num_images` images and denormalize them
    images = images[:num_images]
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    images = images * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
    images = images.clamp(0, 1)  # Ensure pixel values are in range [0, 1]

    # Create a grid of images
    grid = torchvision.utils.make_grid(images, nrow=grid_size[1], padding=2)
    
    # Plot the images
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0))  # Permute to (H, W, C) for matplotlib
    plt.axis('off')
    plt.show()
