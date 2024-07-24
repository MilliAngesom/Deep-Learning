import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the analytical expressions for gradients
def compute_gradients(X, Y, a, b):
    Y_ = a * X + b
    diff = Y_ - Y
    grad_a = 2 * torch.sum(diff * X)
    grad_b = 2 * torch.sum(diff)
    return grad_a, grad_b

# Number of data points
num_points = 50

# Generate random data
X = torch.randn(num_points)
Y = 3 * X + 2 + 0.5 * torch.randn(num_points)

# Define parameters
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Learning rate and optimizer
learning_rate = 0.01
optimizer = optim.SGD([a, b], lr=learning_rate)

# Training loop
num_iterations = 100
losses = []

for i in range(num_iterations):
    # Forward pass
    Y_ = a * X + b

    # Quadratic loss
    loss = torch.sum((Y_ - Y) ** 2)
    losses.append(loss.item())

    # Backward pass
    loss.backward()

    # Explicitly calculate gradients
    grad_a, grad_b = compute_gradients(X, Y, a, b)

    # Print gradients
    print(f'Step: {i}, Loss: {loss.item()}, Gradient (a): {grad_a}, Gradient (b): {grad_b}')
    print(f'Step: {i}, Loss: {loss.item()}, Gradient (a): {a.grad}, Gradient (b): {b.grad}')

    # Compare with PyTorch's gradients
    if torch.isclose(grad_a, a.grad) and torch.isclose(grad_b, b.grad):
        print("Gradients match PyTorch's gradients.")
    else:
        print("Gradients do not match PyTorch's gradients.")

    # Optimization step
    optimizer.step()

    # Reset gradients to zero
    optimizer.zero_grad()


# Visualize the data and the fitted line
plt.scatter(X, Y, label="Data")
plt.plot(X, a.detach().numpy() * X.numpy() + b.detach().numpy(), 'r', label="Fitted Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# Plot the loss curve
plt.plot(range(num_iterations), losses)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()



