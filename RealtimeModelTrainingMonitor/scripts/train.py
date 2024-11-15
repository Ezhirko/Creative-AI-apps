import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import random
from io import BytesIO
import base64
import json

torch.cuda.empty_cache()

# Define CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, kernel_sizes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, kernel_sizes[0], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(kernel_sizes[0], kernel_sizes[1], kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(kernel_sizes[1], kernel_sizes[2], kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(kernel_sizes[2] * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = torch.max_pool2d(torch.relu(self.conv3(x)), 2)
        x = x.view(-1, x.size(1) * 7 * 7)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Define function to train model for one epoch
def train_one_epoch(epoch_no, model, device, train_loader, optimizer, criterion, model_num):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    print('Training started for model:',model_num)
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        torch.cuda.empty_cache()
    
    accuracy = 100. * correct / total
    print(f"Model {model_num}: loss={(epoch_loss / len(train_loader))}, accuracy={accuracy}, epoch={epoch_no}")
    return epoch_loss / len(train_loader), accuracy

def evaluate_random_samples(model, model_name,test_dataset,device):
    model.eval()
    samples = []
    with torch.no_grad():
        indices = random.sample(range(len(test_dataset)), 5)
        for idx in indices:
            image, label = test_dataset[idx]
            image = image.unsqueeze(0).to(device)
            output = model(image)
            pred = output.argmax(dim=1, keepdim=True).item()
            
            # Convert image to base64 for display
            plt.figure(figsize=(1, 1))
            plt.imshow(image.cpu().squeeze(), cmap='gray')
            plt.axis('off')
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            
            img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            samples.append({
                'image': img_data,
                'predicted': pred,
                'actual': label,
                'model': model_name
            })
    return samples

# Save plot and results to HTML
def save_results_html(epoch, losses1, losses2, acc1, acc2):
    plt.figure()
    plt.plot(acc1, label="Model 1 Accuracy")
    plt.plot(acc2, label="Model 2 Accuracy")
    plt.legend()
    plt.savefig("static/accuracy_plot.png")
    
    plt.figure()
    plt.plot(losses1, label="Model 1 Loss")
    plt.plot(losses2, label="Model 2 Loss")
    plt.legend()
    plt.savefig("static/loss_plot.png")

def get_model_summary(model):
    """Get model architecture information"""
    mdl_layers = []
    tlt_params = 0
    # Get layer information
    for name, layer in model.named_children():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            layer_info = {
                'name': name,
                'type': layer.__class__.__name__,
                'shape': list(layer.weight.shape)
            }
            mdl_layers.append(layer_info)
    tlt_params = sum(p.numel() for p in model.parameters())
    layer_result = ''
    for layer in mdl_layers:
        data_str = " ".join([f"{key}: {value}" for key, value in layer.items()])
        layer_result = "<br>".join([layer_result,data_str])

    mdl_summary = {'Layers':layer_result,'Total_Parameters':tlt_params}
    
    return mdl_summary

# Main training loop
def main(kernel1, kernel2, optimizer, batchSize, epochs, learningRate):
    # Prepare datasets and models
    kernel_sizes1 = [int(k) for k in kernel1.split(',')]
    kernel_sizes2 = [int(k) for k in kernel2.split(',')]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batchSize, shuffle=True)

    model1 = SimpleCNN(kernel_sizes1).to(device)
    mdl_summary_1 = get_model_summary(model1)
    print(f"Model 1: layers={mdl_summary_1['Layers']},  tlt_params={mdl_summary_1['Total_Parameters']}")

    model2 = SimpleCNN(kernel_sizes2).to(device)
    mdl_summary_2 = get_model_summary(model2)
    print(f"Model 2: layers={mdl_summary_2['Layers']},  tlt_params={mdl_summary_2['Total_Parameters']}")

    criterion = nn.CrossEntropyLoss()
    if(optimizer == 'Adam'):
        optimizer1 = optim.Adam(model1.parameters(), lr=learningRate)
        optimizer2 = optim.Adam(model2.parameters(), lr=learningRate)
    else:
        optimizer1 = optim.SGD(model1.parameters(), lr=learningRate)
        optimizer2 = optim.SGD(model2.parameters(), lr=learningRate)

    num_epochs = epochs
    losses1, acc1 = [], []
    losses2, acc2 = [], []

    for epoch in range(1, num_epochs + 1):
        # Train Model 1
        loss1, accuracy1 = train_one_epoch(epoch,model1, device, train_loader, optimizer1, criterion,1)
        losses1.append(loss1)
        acc1.append(accuracy1)
        samples1 = evaluate_random_samples(model1,"Model 1",test_data,device)

        # Train Model 2
        loss2, accuracy2 = train_one_epoch(epoch,model2, device, train_loader, optimizer2, criterion,2)
        losses2.append(loss2)
        acc2.append(accuracy2)
        samples2 = evaluate_random_samples(model2,"Model 2",test_data,device)

        # Combine samples
        all_samples = {
            'model1': samples1,
            'model2': samples2
        }
        with open('static/test_samples.json', 'w') as f:
            json.dump(all_samples, f)

        # Save results to HTML after each epoch
        save_results_html(epoch, losses1, losses2, acc1, acc2)

if __name__ == "__main__":
    try:
        kernel1 = sys.argv[1]
        kernel2 = sys.argv[2]
        optimizer = sys.argv[3]
        batchSize = sys.argv[4]
        epochs = sys.argv[5]
        learningRate = sys.argv[6]

        batchSize = int(batchSize)
        epochs = int(epochs)
        learningRate = float(learningRate)

        print(f'From train.py: {kernel1} {kernel2} {optimizer} {batchSize} {epochs} {learningRate}')
        main(kernel1, kernel2, optimizer, batchSize, epochs, learningRate)
    except IndexError:
        print("Error: Expected kernel sizes as arguments.")
