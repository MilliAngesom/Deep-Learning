import os
from pathlib import Path
import pickle
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import skimage as ski
from PIL import Image
import matplotlib.pyplot as plt

def shuffle_data(data_x, data_y):
  indices = np.arange(data_x.shape[0])
  np.random.shuffle(indices)
  shuffled_data_x = np.ascontiguousarray(data_x[indices])
  shuffled_data_y = np.ascontiguousarray(data_y[indices])
  return shuffled_data_x, shuffled_data_y

def unpickle(file):
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding='latin1')
  fo.close()
  return dict

DATA_DIR = '/home/milli/Desktop/Uni_Zagreb/Deep_Learning/lab2/cifar-10-batches-py'
SAVE_DIR = Path(__file__).parent / 'out_CIFAR_SVM_loss'

img_height = 32
img_width = 32
num_channels = 3
num_classes = 10

train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
train_y = []
for i in range(1, 6):
  subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
  train_x = np.vstack((train_x, subset['data']))
  train_y += subset['labels']
train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1)
train_y = np.array(train_y, dtype=np.int32)

subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1).astype(np.float32)
test_y = np.array(subset['labels'], dtype=np.int32)

valid_size = 5000
train_x, train_y = shuffle_data(train_x, train_y)
valid_x = train_x[:valid_size, ...]
valid_y = train_y[:valid_size, ...]
train_x = train_x[valid_size:, ...]
train_y = train_y[valid_size:, ...]
data_mean = train_x.mean((0, 1, 2))
data_std = train_x.std((0, 1, 2))

train_x = (train_x - data_mean) / data_std
valid_x = (valid_x - data_mean) / data_std
test_x = (test_x - data_mean) / data_std

train_x = train_x.transpose(0, 3, 1, 2)
valid_x = valid_x.transpose(0, 3, 1, 2)
test_x = test_x.transpose(0, 3, 1, 2)




# CNN Model
# class ConvolutionalModel(nn.Module):
#     def __init__(self):
#         super(ConvolutionalModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, 5)  
#         self.pool = nn.MaxPool2d(3, 2)  # output size from the pooliing layer [floor( (input size - filter size) / stride ) + 1 ]
#         self.conv2 = nn.Conv2d(16, 32, 5)
#         self.fc1 = nn.Linear(32 * 4 * 4, 256)  
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 10)

#     def forward(self, x):
#         # print("conv1 output",self.conv1(x).shape)
#         x = self.pool(F.relu(self.conv1(x)))
#         # print("pool1 output",x.shape)
#         # print("conv2 output",self.conv2(x).shape)
#         x = self.pool(F.relu(self.conv2(x)))
#         # print("pool2 output",x.shape)
#         x = x.view(-1, 32 * 4 * 4)  # Flatten the tensor for the fully connected layer
#         # print("flatten output", x.shape)
#         x = F.relu(self.fc1(x))
#         # print("FC1 output",x.shape)
#         x = F.relu(self.fc2(x))
#         # print("FC2 output",x.shape)
#         x = self.fc3(x)
#         # print("FC3 output",x.shape)
#         return x
    
####################################################################################################################
# improved CNN
class ConvolutionalModel(nn.Module):
    def __init__(self):
        super(ConvolutionalModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)  # output size from the pooling layer [floor( (input size - filter size) / stride ) + 1 ]
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
#######################################################################################################################

# Evaluation Function
def evaluate(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    # Calculate the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    # Calculate precision, recall, and f-score for each class
    precision, recall, fscore, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)

    # Calculate overall precision, recall, and f-score
    overall_precision = np.mean(precision)
    overall_recall = np.mean(recall)
    overall_fscore = np.mean(fscore)

    print('Confusion Matrix:\n', cm)
    print('Class-wise Precision:', precision)
    print('Class-wise Recall:', recall)
    print('Class-wise F-score:', fscore)
    print('Overall Precision:', overall_precision)
    print('Overall Recall:', overall_recall)
    print('Overall F-score:', overall_fscore)
    # print('Accuracy:', np.mean(np.array(all_labels) == np.array(all_preds)))
    return np.mean(np.array(all_labels) == np.array(all_preds))








def exponential_decay_lr(epoch, initial_lr, decay_rate):
    # Assuming exponential decay without a staircase effect
    lr = initial_lr * np.exp(-decay_rate * epoch)
    return lr

def plot_metrics(metrics, title, ylabel):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    for label, values in metrics.items():
        plt.plot(values, label=label)
    plt.legend()
    plt.show()

# Function to adjust learning rate
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def draw_conv_filters(epoch, step, weights, save_dir):
  w = weights.copy()
  num_filters = w.shape[0]
  num_channels = w.shape[1]
  k = w.shape[2]
  assert w.shape[3] == w.shape[2]
  w = w.transpose(2, 3, 1, 0)
  w -= w.min()
  w /= w.max()
  border = 1
  cols = 8
  rows = math.ceil(num_filters / cols)
  width = cols * k + (cols-1) * border
  height = rows * k + (rows-1) * border
  img = np.zeros([height, width, num_channels])
  for i in range(num_filters):
    r = int(i / cols) * (k + border)
    c = int(i % cols) * (k + border)
    img[r:r+k,c:c+k,:] = w[:,:,:,i]
#   filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
#   ski.io.imsave(os.path.join(save_dir, filename), img)

  # Resizing the image
    zoom = 10
    img_resized = Image.fromarray((img*255).astype(np.uint8))
    img_resized = img_resized.resize((width * zoom, height * zoom), Image.NEAREST)

    filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
    # img_resized.save(os.path.join(save_dir, filename))

def multiclass_hinge_loss(logits: torch.Tensor, target: torch.Tensor, delta=1.):
    """
    Args:
        logits: torch.Tensor with shape (B, C), where B is batch size, and C is number of classes.
        target: torch.LongTensor with shape (B, ) representing ground truth labels.
        delta: Hyperparameter.
    Returns:
        Loss as scalar torch.Tensor.
    """
    # Number of classes
    num_classes = logits.size(1)
    
    # Create a mask for the correct class logits
    correct_class_mask = torch.nn.functional.one_hot(target, num_classes).bool()
    
    # Select the correct class logits
    correct_class_logits = torch.masked_select(logits, correct_class_mask)
    
    # Reshape for broadcasting
    correct_class_logits = correct_class_logits.view(-1, 1)
    
    # Calculate differences between correct class logits and all other logits
    differences = delta + logits - correct_class_logits
    
    # Apply the mask to zero-out differences for correct classes
    differences[correct_class_mask] = 0
    
    # Calculate hinge loss
    hinge_losses = torch.max(differences, torch.zeros_like(differences))
    
    # Sum over all incorrect classes, and average over the batch
    loss = torch.sum(hinge_losses) / logits.size(0)
    
    return loss

def train_model(model, train_x, train_y, valid_x, valid_y, num_epochs=50, batch_size=64, initial_lr=0.01, decay_rate=0.05):
    # Convert numpy arrays to torch tensors
    train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
    train_y_tensor = torch.tensor(train_y, dtype=torch.long)
    valid_x_tensor = torch.tensor(valid_x, dtype=torch.float32)
    valid_y_tensor = torch.tensor(valid_y, dtype=torch.long)

    # Create data loaders
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = TensorDataset(valid_x_tensor, valid_y_tensor)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    # Define the loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    lr = initial_lr
    # Metrics to plot
    metrics = {
        'train_loss': [],
        'learning_rate': [],
        'training_accuracy': [],
        'validation_accuracy': [],
        'validation_loss': []
    }

    draw_conv_filters(0, 0, model.conv1.weight.detach().numpy(), SAVE_DIR)
    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            # loss = criterion(outputs, labels)
            loss = multiclass_hinge_loss(outputs, labels, delta=1.)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        # training_accuracy = correct / total  # if the accuracy is to be computed here and not using teh evaluate function----> uncomment this line
        

        # Evaluation on the training set
        train_accuracy = evaluate(model, train_loader)
        
        # assert train_accuracy== training_accuracy
        # Evaluation on the validation set
        validation_accuracy = evaluate(model, valid_loader)


        # to compute for the validation loss
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                # Forward pass
                outputs = model(inputs)
                # loss = criterion(outputs, labels)
                loss = multiclass_hinge_loss(outputs, labels, delta=1.)

                # Statistics
                running_loss += loss.item()
            valid_loss = running_loss / len(valid_loader)

        # Record metrics
        metrics['train_loss'].append(train_loss)
        metrics['learning_rate'].append(lr)
        metrics['training_accuracy'].append(train_accuracy)
        metrics['validation_accuracy'].append(validation_accuracy)
        metrics['validation_loss'].append(valid_loss)

        # Print statistics
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss}, LR: {lr}, Training Acc: {train_accuracy}')
        
        # Update the learning rate
        lr = exponential_decay_lr(epoch, initial_lr, decay_rate)
        adjust_learning_rate(optimizer, lr)
        draw_conv_filters(epoch, 1, model.conv1.weight.detach().numpy(), SAVE_DIR)

    # Plot metrics
    plot_metrics({
        'Training Loss': metrics['train_loss'],
        'Validation Loss': metrics['validation_loss']
    }, "Loss Over Time", "Loss")

   

    plot_metrics({'Learning Rate': metrics['learning_rate']}, "Learning Rate Over Time", "Learning Rate")
    plot_metrics({
        'Training Accuracy': metrics['training_accuracy'],
        'Validation Accuracy': metrics['validation_accuracy']
    }, "Accuracy Over Time", "Accuracy")

    return model


# Initialize the model
model = ConvolutionalModel()

# Load the state dictionary
# model.load_state_dict(torch.load('model_weights.pth'))
model.load_state_dict(torch.load('model_weights_SVM_loss.pth'))

# Train the model
# model = train_model(model, train_x, train_y, valid_x, valid_y, num_epochs=50, batch_size=50, initial_lr=0.01)

# Save the model's state dictionary
# torch.save(model.state_dict(), 'model_weights.pth')
# torch.save(model.state_dict(), 'model_weights_SVM_loss.pth') 

# Evaluating the model on the test dataset
print("Evaluating the model on the test dataset")

# Convert test data to tensors and create a DataLoader
test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
test_y_tensor = torch.tensor(test_y, dtype=torch.long)
test_dataset = TensorDataset(test_x_tensor, test_y_tensor)
test_loader = DataLoader(test_dataset, batch_size=500)

model.eval()
all_preds = []
all_labels = []
losses = []
incorrect_samples = []
# Use criterion with reduction set to 'none' to get individual losses
criterion = torch.nn.CrossEntropyLoss(reduction='none')

# Iterate over the test data
for inputs, labels in test_loader:
    outputs = model(inputs)
    losses = criterion(outputs, labels)  # This will now give a tensor of losses for each image
    # losses = multiclass_hinge_loss(outputs, labels, delta=1.)

    _, predicted = torch.max(outputs, 1)
    all_preds.extend(predicted.numpy())
    all_labels.extend(labels.numpy())

    # Store incorrectly classified samples with their individual loss
    for i in range(len(inputs)):
        if predicted[i] != labels[i]:
            incorrect_samples.append((inputs[i], predicted[i], labels[i], losses[i].item()))

# Now, sort the incorrect samples by individual loss
top_incorrect_samples = sorted(incorrect_samples, key=lambda x: x[3], reverse=True)[:20]

# Generate confusion matrix and calculate class-wise accuracy
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)
total_accuracy = np.trace(cm) / np.sum(cm)
print("Test Accuracy: ",total_accuracy)
class_accuracies = cm.diagonal() / cm.sum(axis=1)
top_classes = np.argsort(class_accuracies)[::-1][:3]

# CIFAR-10 classes
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
# Visualization
data_mean = torch.tensor(data_mean).view(3, 1, 1)
data_std = torch.tensor(data_std).view(3, 1, 1)
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
fig.suptitle('Top 20 Incorrectly Classified Images with Largest Loss')
for i, ax in enumerate(axes.flatten()):
    img, pred, actual, _ = top_incorrect_samples[i]
    img = img * data_std + data_mean  # Undo normalization
    img = img.permute(1, 2, 0).numpy()  # Rearrange dimensions for plotting
    ax.imshow(img.astype(np.uint8))
    ax.set_title(f'Actual: {class_names[actual]}, Predicted: {class_names[pred]}')
    ax.axis('off')

plt.show()

print("Top 3 best-performing classes:\n",top_classes[0],": ", class_names[top_classes[0]])
print(top_classes[1],": ", class_names[top_classes[1]])
print(top_classes[2],": ", class_names[top_classes[2]])
