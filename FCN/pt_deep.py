import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from DATA import *

class PTDeep(nn.Module):
    def __init__(self, config, activation):
        super(PTDeep, self).__init__()
        self.config = config
        self.activation = activation

        # Create lists to store weights and biases for each layer
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        # Initialize weights and biases for each layer
        for i in range(len(config) - 1):
            self.weights.append(nn.Parameter(torch.randn(config[i], config[i + 1])))
            self.biases.append(nn.Parameter(torch.zeros(1, config[i + 1])))

    def forward(self, X):
        h = X
        h = h.unsqueeze(0) if h.dim() == 1 else h
        for i in range(len(self.weights) - 1):
            h = self.activation(torch.mm(h, self.weights[i]) + self.biases[i])
        
        logits = torch.mm(h, self.weights[-1]) + self.biases[-1]
        max_logits = torch.max(logits, dim=1, keepdim=True).values
        probs = torch.softmax(logits - max_logits, dim=1)
        return probs

    
    
    def get_loss(self, X, Yoh_):
      regularization_coeff = 0.001
      epsilon =1e-10
      probs = self(X) + epsilon # to make sure we don't enocunter with log(0)
      data_loss = -torch.sum(Yoh_ * torch.log(probs)) / X.size(0)
      regularization_loss = 0.5 * regularization_coeff * sum([torch.sum(w**2) for w in self.weights] + [torch.sum(b**2) for b in self.biases])
      loss_ = data_loss + regularization_loss
      return loss_


    def count_params(self):
        total_params = 0
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            print(f"Layer {i+1}:")
            print(f"Weights shape: {weight.shape}")
            print(f"Biases shape: {bias.shape}")
            total_params += weight.numel() + bias.numel()
        print(f"Total number of parameters: {total_params}")

# Define functions for training and evaluation USING Batch Gradient Descent
def train(model, X, Yoh_, param_niter, param_delta):
    global losses
    optimizer = optim.SGD(model.parameters(), lr=param_delta)

    losses = []
    for i in range(param_niter):
        optimizer.zero_grad()
        loss = model.get_loss(X, Yoh_)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (i + 1) % 100 == 0:
            print(f'Step {i + 1}, Loss: {loss.item()}')
        
# # Define functions for training and evaluation USING SGD
# def train(model, X, Yoh_, param_niter, param_delta):
#     global losses
#     optimizer = optim.SGD(model.parameters(), lr=param_delta)
    
#     losses = []
#     for i in range(param_niter):
#         for j in range(len(X)):
#             optimizer.zero_grad()
#             loss = model.get_loss(X[j], Yoh_[j])
#             loss.backward()
#             optimizer.step()
#             losses.append(loss.item())

#         if (i + 1) % 100 == 0:
#             print(f'Step {i + 1}, Loss: {loss.item()}')



def eval(model, X):
    with torch.no_grad():
        probs = model(X)
    return probs

def display_decision_boundary(X, Yoh_, model):
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model(torch.Tensor(grid_points)).detach().numpy()
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Yoh_, cmap=plt.cm.Spectral, edgecolor='k')
    plt.show()

def calculate_precision_recall(Y_true, predictions, num_classes):
    precision = torch.zeros(num_classes)
    recall = torch.zeros(num_classes)

    for c in range(num_classes):
        true_positive = torch.sum((predictions == c) & (Y_true == c)).item()
        false_positive = torch.sum((predictions == c) & (Y_true != c)).item()
        false_negative = torch.sum((predictions != c) & (Y_true == c)).item()

        precision[c] = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall[c] = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0

    return precision, recall


if __name__ == "__main__":
    global losses
    np.random.seed(750)

    # Generate data
    K = 6
    C = 2
    N = 10
    X, Y_ = sample_gmm_2d(K, C, N)
    
    # Convert data to torch.Tensor
    X = torch.Tensor(X)
    Yoh_ = torch.Tensor(class_to_onehot(Y_))
    
    # Define the model with configuration [2, 3]
    ptdeep = PTDeep([2,10,2], activation=torch.relu)  # use torch.relu or torch.sigmoid

    # Train the model
    num_iterations = 100000
    train(ptdeep, X, Yoh_, num_iterations, 0.01)

    # Get probabilities on training data
    probs = eval(ptdeep, X)

    # Calculate performance metrics
    predictions = torch.argmax(probs, dim=1)
    Y_true = torch.argmax(Yoh_, dim=1)
    accuracy = torch.sum(predictions == Y_true).item() / (N * K)

    print(f"Accuracy: {accuracy}")

    # Calculate precision and recall
    num_classes = C
    precision, recall = calculate_precision_recall(Y_true, predictions, num_classes)

    for c in range(num_classes):
        print(f"Class {c} - Precision: {precision[c]:.4f}, Recall: {recall[c]:.4f}")

    # Display the decision boundary
    display_decision_boundary(X, Y_, ptdeep)

    # Count and print the model's parameters
    ptdeep.count_params()
    
    # Plot the loss curve
    # plt.plot(range(len(X)*num_iterations), losses) # for SGD
    # plt.title("SGD")
    plt.plot(range(num_iterations), losses) # for Batch Greadient Descent
    plt.title("Batch Gradient Descent")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()
