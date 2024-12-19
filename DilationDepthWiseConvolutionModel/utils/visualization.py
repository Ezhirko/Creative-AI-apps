# visualization.py
import matplotlib.pyplot as plt
import numpy as np
import torch

# CIFAR-10 class labels
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer", "dog", 
    "frog", "horse", "ship", "truck"
]

def imshow(img):
    """Helper function to unnormalize and display the image"""
    img = img / 2 + 0.5  # Unnormalize
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.axis('off')  # Hide axes

def visualize_samples(data_loader, num_samples=25):
    """Visualizes 25 sample images from the dataset in a 5x5 grid."""
    # Get a batch of images
    data_iter = iter(data_loader)
    images, labels = next(data_iter)

    # Set up the matplotlib figure
    plt.figure(figsize=(5, 5))
    
    # Display the first 25 images
    for i in range(num_samples):
        ax = plt.subplot(5, 5, i + 1)
        imshow(images[i])
        label_name = CIFAR10_CLASSES[labels[i].item()]  # Convert integer label to text
        ax.set_title(f"{label_name}", fontsize=8)

    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    plt.show()

def visualize_loss_accuracy_plot(train_losses,test_losses,train_acc,test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    # Move tensors to CPU and convert to NumPy
    train_losses_cpu = [loss.cpu().item() if torch.is_tensor(loss) else loss for loss in train_losses]
    train_acc_cpu = [acc.cpu().item() if torch.is_tensor(acc) else acc for acc in train_acc]
    test_losses_cpu = [loss.cpu().item() if torch.is_tensor(loss) else loss for loss in test_losses]
    test_acc_cpu = [acc.cpu().item() if torch.is_tensor(acc) else acc for acc in test_acc]

    axs[0, 0].plot(train_losses_cpu)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc_cpu)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses_cpu)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc_cpu)
    axs[1, 1].set_title("Test Accuracy")
