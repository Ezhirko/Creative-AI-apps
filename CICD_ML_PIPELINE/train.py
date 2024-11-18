import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchsummary as summary
import torch.nn.functional as F

# Define the model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(6)  # Batch normalization
        self.conv2 = nn.Conv2d(6, 6, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(6)  # Batch normalization
        self.fc1 = nn.Linear(6 * 7 * 7, 80)
        self.fc2 = nn.Linear(80, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 6 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training script
def train():
    # Transform and load data
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Model, loss, optimizer
    model = SimpleCNN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameter:{total_params}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    print(f"loss={(epoch_loss / len(train_loader))}, accuracy={accuracy}")

    # Save the model with a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), f"model_{timestamp}.pth")

    print("Model trained and saved.")
    return accuracy


if __name__ == "__main__":
    train()
