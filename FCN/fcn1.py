import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from DATA import * 

# Batch normalisation implemented using numpy 
class PTDeep(nn.Module):
    def __init__(self, config, activation_fn, momentum=0.1):
        super(PTDeep, self).__init__()
        self.config = config
        self.activation_fn = activation_fn
        self.momentum = momentum

        # Create lists to store layers and batch norm parameters
        self.layers = nn.ModuleList()
        self.gammas = nn.ParameterList()
        self.betas = nn.ParameterList()
        self.running_means = []
        self.running_vars = []

        # Initialize layers and batch norm parameters
        for i in range(len(config) - 2):
            self.layers.append(nn.Linear(config[i], config[i + 1]))
            self.gammas.append(nn.Parameter(torch.ones(config[i + 1])))
            self.betas.append(nn.Parameter(torch.zeros(config[i + 1])))
            self.layers.append(activation_fn())
            self.running_means.append(nn.Parameter(torch.zeros(config[i + 1]), requires_grad=False))
            self.running_vars.append(nn.Parameter(torch.ones(config[i + 1]), requires_grad=False))

        # Add the output layer
        self.layers.append(nn.Linear(config[-2], config[-1]))

    def custom_batch_norm(self, x, gamma, beta, running_mean, running_var, is_training, epsilon=1e-5):
        if is_training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            running_mean.data = (1 - self.momentum) * running_mean.data + self.momentum * batch_mean.data
            running_var.data = (1 - self.momentum) * running_var.data + self.momentum * batch_var.data
            normalized = (x - batch_mean) / torch.sqrt(batch_var + epsilon)
        else:
            normalized = (x - running_mean) / torch.sqrt(running_var + epsilon)
        return gamma * normalized + beta

    def forward(self, x):
        is_training = self.training
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                x = layer(x)
                if i < len(self.layers) - 2:  # Apply custom batch norm
                    gamma = self.gammas[i // 2]
                    beta = self.betas[i // 2]
                    running_mean = self.running_means[i // 2]
                    running_var = self.running_vars[i // 2]
                    x = self.custom_batch_norm(x, gamma, beta, running_mean, running_var, is_training)
            else:
                x = layer(x)
        return x

    def get_loss(self, X, Yoh_):
        regularization_coeff = 0.0001
        logits = self.forward(X)
        data_loss = F.cross_entropy(logits, Yoh_.max(1)[1])
        
        regularization_loss = 0.5 * regularization_coeff * sum(
            [torch.sum(w.weight ** 2) for w in self.layers if isinstance(w, nn.Linear)] +
            [torch.sum(w.bias ** 2) for w in self.layers if isinstance(w, nn.Linear)]
        )
        
        loss_ = data_loss + regularization_loss
        return loss_

    def count_params(self):
        total_params = 0
        for layer in self.layers:
            if hasattr(layer, 'weight'):
                total_params += layer.weight.numel()
            if hasattr(layer, 'bias'):
                total_params += layer.bias.numel()
        print(f"Total number of parameters: {total_params}")


# Define functions for training and evaluation
def train(model, X, Yoh_, param_niter, param_delta):
    optimizer = optim.SGD(model.parameters(), lr=param_delta)
    
    for i in range(param_niter):
        model.train()  # Set the model to training mode
        optimizer.zero_grad()
        loss = model.get_loss(X, Yoh_)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Step {i + 1}, Loss: {loss.item()}')
    

def eval(model, X):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        probs = model(X)
    return probs

def display_decision_boundary(X, Y_, model):
    # Set the model to evaluation mode for consistent prediction
    model.eval()
    
    # Generate a grid of points over the data range
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Flatten the grid so the values match the expected input format
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid = torch.Tensor(grid)

    # Predict the function value for the whole grid
    Z = model(grid)
    Z = Z.argmax(1).numpy()
    Z = Z.reshape(xx.shape)

    # Use a colormap that offers clear contrast
    cmap = plt.cm.coolwarm

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap)
    
    # Draw the decision boundaries
    plt.contour(xx, yy, Z, colors='k', linewidths=2, linestyles='solid')

    # Overlay the data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=Y_, edgecolors='k', s=50, cmap=cmap)

    # Create a legend
    plt.legend(*scatter.legend_elements(), title="Classes")

    # Show the plot
    plt.show()
    

# Sample test code
if __name__ == "__main__":
    np.random.seed(100)
    torch.manual_seed(100)

    # Generate data
    K = 6
    C = 2
    N = 10
    X, Y_ = sample_gmm_2d(K, C, N)

    # Convert data to torch.Tensor
    X = torch.Tensor(X)
    Yoh_ = torch.Tensor(class_to_onehot(Y_))

    # Define the model with configuration [2, 10, 10, 2] and sigmoid activation
    ptdeep = PTDeep([2,10,10,10,2], activation_fn=nn.Sigmoid)  

    # Train the model
    train(ptdeep, X, Yoh_, param_niter=10000, param_delta=0.01)

    # Evaluate the model
    probs = eval(ptdeep, X)
    predictions = torch.argmax(probs, dim=1)
    accuracy = torch.sum(predictions == torch.argmax(Yoh_, dim=1)).item() / (N * K)
    
    print(f"Accuracy: {accuracy}")

    # Count and print the model's parameters
    ptdeep.count_params()

    # Display the decision boundary
    display_decision_boundary(X.numpy(), Y_, ptdeep)