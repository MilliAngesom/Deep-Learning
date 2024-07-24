import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from DATA import *

# Define the class PTLogreg for logistic regression
class PTLogreg(nn.Module):
    def __init__(self, D, C):
        super(PTLogreg, self).__init__()
        self.W = nn.Parameter(torch.randn(D, C))
        self.b = nn.Parameter(torch.zeros(C))

    def forward(self, X):
        scores = torch.mm(X, self.W) + self.b
        probs = torch.softmax(scores, dim=1)
        return probs

    def get_loss(self, X, Yoh_):
        regularization_coeff = 0.01
        probs = self(X)
        data_loss= -torch.sum(Yoh_ * torch.log(probs)) / X.size(0)
        regularization_loss = 0.5 * regularization_coeff * (torch.sum(self.W**2) + torch.sum(self.b**2))
        loss = data_loss + regularization_loss
        return loss

# Define functions for training and evaluation
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
    np.random.seed(110)

    # Generate data (You can modify this part as needed)
    K = 4
    C = 3
    N = 30
    X, Y_ = sample_gmm_2d(K, C, N)
    

    # Convert data to torch.Tensor
    X = torch.Tensor(X)
    Yoh_ = torch.Tensor(class_to_onehot(Y_))
    

    # Define the model
    ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])

    # Train the model
    num_iterations = 1000
    train(ptlr, X, Yoh_, num_iterations, 0.005)

    # Get probabilities on training data
    probs = eval(ptlr, X)

    # Calculate performance metrics
    predictions = torch.argmax(probs, dim=1)
    Y_true = torch.argmax(Yoh_, dim=1)
    accuracy = torch.sum(predictions == Y_true).item() / (N*K)

    print(f"Accuracy: {accuracy}")

    # Calculate precision and recall
    num_classes = C
    precision, recall = calculate_precision_recall(Y_true, predictions, num_classes)

    for c in range(num_classes):
        print(f"Class {c} - Precision: {precision[c]:.4f}, Recall: {recall[c]:.4f}")

    
    # Display the decision boundary
    display_decision_boundary(X, Y_, ptlr)

    # Plot the loss curve
    plt.plot(range(num_iterations), losses) 
    plt.title("Batch Gradient Descent")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

''' above is the implementation using batch gradient descent'''