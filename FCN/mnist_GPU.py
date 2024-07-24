import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_root = '/home/milli'  # Change this to your preference
mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=False)
mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=False)

x_train, y_train = mnist_train.data, mnist_train.targets
x_test, y_test = mnist_test.data, mnist_test.targets
x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

N = x_train.shape[0]
D = x_train.shape[1] * x_train.shape[2]
C = y_train.max().add_(1).item()

# Flatten the images
x_train = x_train.view(N, D)
x_test = x_test.view(-1, D)

class PTDeep(nn.Module):
    def __init__(self, dims):
        super(PTDeep, self).__init__()
        self.dims = dims
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x

    def predict(self, x):
        logits = self.forward(x)
        return logits.argmax(dim=1)

def train(model, x_train_full, y_train_full, x_test, y_test, epochs, lr, reg, patience=10):
    model.to(device)  # Move model to GPU if available

    x_train_full, y_train_full = x_train_full.to(device), y_train_full.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

    best_val_loss = float('inf')
    counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train_full)
        loss = criterion(outputs, y_train_full)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_outputs = model(x_test)
            test_loss = criterion(test_outputs, y_test)

        if test_loss < best_val_loss:
            best_val_loss = test_loss
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            break

    model.load_state_dict(best_model_state)

def train_mb(model, x_train_full, y_train_full, x_test, y_test, epochs, lr, reg, batch_size, patience=10):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    
    best_val_loss = float('inf')
    counter = 0

    # Convert the full dataset to GPU if available
    x_train_full, y_train_full = x_train_full.to(device), y_train_full.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    for epoch in range(epochs):
        model.train()

        permutation = torch.randperm(x_train_full.size()[0])
        for i in range(0, x_train_full.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = x_train_full[indices], y_train_full[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_outputs = model(x_test)
            test_loss = criterion(test_outputs, y_test)

        if test_loss < best_val_loss:
            best_val_loss = test_loss
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break

    model.load_state_dict(best_model_state)

def train_svm_classifier(x_train, y_train, x_test, y_test, kernel_type='linear'):
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    svm_classifier = make_pipeline(StandardScaler(), SVC(kernel=kernel_type))
    svm_classifier.fit(x_train, y_train)
    predictions = svm_classifier.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy


def main():
    dims = [D, C]  # Example: Input layer (D), Hidden layers (100, 100), Output layer (C)
    
    model = PTDeep(dims).to(device)
    train_mb(model, x_train, y_train, x_test, y_test, epochs=100, lr=0.0001, reg=0.000, batch_size=10)

if __name__ == "__main__":
    main()
