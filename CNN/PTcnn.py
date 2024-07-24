import os
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from PIL import Image
import numpy as np
from pathlib import Path

# CNN model

"""
    Initialize the Convolutional Neural Network.

    Args:
    in_channels (int): Number of channels in the input image. For example, 1 for grayscale and 3 for RGB images.
    conv1_width (int): Number of filters (output channels) in the first convolutional layer.
    conv2_width (int): Number of filters (output channels) in the second convolutional layer.
    fc1_width (int): Number of neurons in the first fully connected (dense) layer.
    class_count (int): Number of output classes, corresponding to the number of neurons in the final layer.

    The architecture includes two convolutional layers each followed by max-pooling, 
    and two fully connected layers. The first and second convolutional layers use the specified 
    numbers of filters. The first fully connected layer has a width as specified, and the final 
    layer's width is determined by the number of classes (class_count). This setup is typical 
    for image classification tasks.
    """
class ConvolutionalModel(nn.Module):
    def __init__(self, in_channels, conv1_width, conv2_width, fc1_width, class_count):
        super(ConvolutionalModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(conv1_width, conv2_width, kernel_size=5, stride=1, padding=2, bias=True)
        self.fc1 = nn.Linear(conv2_width * 7 * 7, fc1_width, bias=True)
        self.fc_logits = nn.Linear(fc1_width, class_count, bias=True)

    def forward(self, x):
        # x = self.pool(torch.relu(self.conv1(x)))
        x = torch.relu(self.pool(self.conv1(x)))
        # x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.pool(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc_logits(x)
        return x

# Function to adjust learning rate
def adjust_learning_rate(optimizer, epoch, lr_policy):
    if epoch in lr_policy:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_policy[epoch]

# Function to visualize and save filters
def visualize_and_save_filters(layer, epoch, step, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filters = layer.weight.data.clone()
    num_filters = filters.shape[0]
    C = filters.shape[1]
    k = filters.shape[2]
    filters = filters - filters.min()
    filters = filters / filters.max()
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols - 1)
    height = rows * k + (rows - 1)
    for i in range(1):
        img = torch.zeros(height, width)
        for j in range(num_filters):
            r = int(j / cols) * (k + 1)
            c = int(j % cols) * (k + 1)
            img[r:r+k, c:c+k] = filters[j, i]
        img = img.numpy()
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        # resize the image
        zoom = 10
        img = img.resize((width * zoom, height * zoom), Image.NEAREST)
        # img.save(os.path.join(save_dir, f'filter_epoch_{epoch}_step_{step}_input_{i}.png'))

# Main training function
def train_and_test(config):
    # Load MNIST dataset
    DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
    

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset_full = datasets.MNIST(DATA_DIR, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=False, transform=transform)

    # Split training dataset into train and validation
    train_dataset = Subset(train_dataset_full, range(55000))
    valid_dataset = Subset(train_dataset_full, range(55000, 60000))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Model, loss, optimizer, and TensorBoard writer
    model = ConvolutionalModel(1, 16, 32, 512, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr_policy'][1], weight_decay=config['weight_decay'])
    writer = SummaryWriter(config['save_dir'])

    # Training loop
    for epoch in range(1, config['max_epochs'] + 1):
        adjust_learning_rate(optimizer, epoch, config['lr_policy'])
        model.train()

        # To accumulate the training loss for each epoch
        total_loss = 0
        total_correct = 0
        total_samples = 0
        # iterate through each mini-batch of the train dataset
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).float().sum().item()
            total_correct += correct
            total_samples += labels.size(0)

            loss.backward()
            optimizer.step()

            if i % 5 == 0:
                # print(f"Epoch {epoch}, Batch {i}, Training Accuracy: {accuracy:.2f}")
                print(f"Epoch {epoch}, Batch {i}, Training Loss: {loss.item():.2f}")

        # Save the learned filters of the first Conv. layer
        visualize_and_save_filters(model.conv1, epoch, i*config['batch_size'], config['save_dir'])

        # Calculate and log average training accuracy and loss for the epoch
        epoch_accuracy = 100. * total_correct / total_samples
        epoch_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}, Training Accuracy: {epoch_accuracy:.2f}%, Training Loss: {epoch_loss:.2f}")
        writer.add_scalar('Accuracy/train', epoch_accuracy, epoch)
        writer.add_scalar('Loss/train', epoch_loss, epoch)

        # Validation after each epoch
        model.eval()
        valid_loss, valid_correct, valid_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).float().sum().item()
            valid_accuracy = 100. * valid_correct / valid_total
            print(f"Epoch {epoch}, Validation Accuracy: {valid_accuracy:.2f}%")
            print(f"Epoch {epoch}, Validation Loss: {valid_loss / valid_total:.2f}")
            writer.add_scalar('Accuracy/validation', valid_accuracy, epoch)
            writer.add_scalar('Loss/validation', valid_loss / valid_total, epoch)

    # Test the model
    model.eval()
    test_correct, test_total, test_incorrect= 0, 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_incorrect += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).float().sum().item()
        test_accuracy = 100. * test_correct / test_total
        test_loss = test_incorrect / test_total
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print(f"Test Loss: {test_loss:.2f}")
        # writer.add_scalar('Accuracy/test', test_accuracy, 1)

    # Close TensorBoard writer
    writer.close()

def plotting_function(config):

    event_file = config['save_dir']

    # Create an accumulator and reload it
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    # Assuming scalar data is stored under tag 'Accuracy/train'
    # Extract scalar data for the tag
    train_accuracy_data = event_acc.Scalars('Accuracy/train')
    train_loss_data = event_acc.Scalars('Loss/train')

    validation_accuracy_data = event_acc.Scalars('Accuracy/validation')
    validation_loss_data = event_acc.Scalars('Loss/validation')

    # Extract values and steps
    steps = [data.step for data in train_accuracy_data]
    train_accuracy = [data.value/100 for data in train_accuracy_data]
    train_loss = [data.value for data in train_loss_data]

    validation_accuracy = [data.value/100 for data in validation_accuracy_data]
    validation_loss = [data.value for data in validation_loss_data]
    
    plt.figure(figsize=(10, 6))  # Optional: you can define the size of the figure

    plt.plot(steps, train_accuracy, label='Training Accuracy')
    plt.plot(steps, train_loss, label='Training Loss')
    plt.plot(steps, validation_accuracy, label='Validation Accuracy')
    plt.plot(steps, validation_loss, label='Validation Loss')

    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.title('Training and Validation Metrics')
    plt.legend()
    plt.grid(True)
    plt.show()

# Configuration
config = {
    'max_epochs': 8,
    'batch_size': 50,
    'save_dir': './task3/PT_unregularised',
    'lr_policy': {1: 1e-1, 3: 1e-2, 5: 1e-3, 7: 1e-4},
    'weight_decay': 0  
}
#'save_dir': './task3/PT_unregularised'

# Train the model
# train_and_test(config)

# Plot the evolution of Train, Validation accuracies and losses
plotting_function(config)



"""
[weight_decay = 1e-3]
    Validation accuracy = 99.18%
    Validation avg loss = 0.00 -----> loss is 0 because the resolution used to save the average loss is 2 digits decimal point 
                                      if this is increased we will have non-zero loss
    Test accuracy = 99.26%
    Test avg loss = 0.00

    
[weight_decay = 1e-2]
    Validation accuracy = 98.86%
    Validation avg loss = 0.00

    Test accuracy = 98.61%
    Test avg loss = 0.00


[weight_decay = 1e-1]
    Validation accuracy = 96.14%
    Validation avg loss = 0.00

    Test accuracy = 95.32%
    Test avg loss = 0.00


[unregularised]
    Validation accuracy = 99.18%
    Validation avg loss = 0.00

    Test accuracy = 99.24%
    Test avg loss = 0.00
"""




