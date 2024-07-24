import numpy as np
from DATA import sample_gmm_2d

class FCANN2:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate, regularization_coeff):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.regularization_coeff = regularization_coeff

        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros((1, output_dim))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)


    def fcann2_train(self, X, Y, num_iterations, batch_size=1): # changing the value of batch_size yields SGD(1), mini-batch(2-59) or batch gradient descent(60)
        m = X.shape[0]

        for i in range(num_iterations):
            # Shuffle dataset
            indices = np.arange(m)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            for j in range(0, m, batch_size):
                # Get the mini-batch
                X_batch = X_shuffled[j:j + batch_size]
                Y_batch = Y_shuffled[j:j + batch_size]

                # Forward propagation 
                Z1 = np.dot(X_batch, self.W1) + self.b1
                A1 = self.sigmoid(Z1)
                Z2 = np.dot(A1, self.W2) + self.b2
                Y_hat = self.softmax(Z2)

                # Compute loss 
                loglikelihood_loss = -np.log(Y_hat[range(len(Y_batch)), Y_batch])
                data_loss = np.sum(loglikelihood_loss) / len(Y_batch)
                regularization_loss = 0.5 * self.regularization_coeff * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
                loss = data_loss + regularization_loss

                # Backpropagation 
                dZ2 = Y_hat
                dZ2[range(len(Y_batch)), Y_batch] -= 1
                dW2 = np.dot(A1.T, dZ2)
                db2 = np.sum(dZ2, axis=0)
                dA1 = np.dot(dZ2, self.W2.T)
                dZ1 = dA1 * A1 * (1 - A1)
                dW1 = np.dot(X_batch.T, dZ1)
                db1 = np.sum(dZ1, axis=0)

                # Update weights and biases 
                self.W1 -= self.learning_rate * (dW1 + self.regularization_coeff * self.W1)
                self.b1 -= self.learning_rate * db1
                self.W2 -= self.learning_rate * (dW2 + self.regularization_coeff * self.W2)
                self.b2 -= self.learning_rate * db2

            if i % 1000 == 0:
                print(f"Iteration {i}, Loss: {loss:.6f}")

    def fcann2_classify(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        Y_hat = self.softmax(Z2)
        return np.argmax(Y_hat, axis=1)


def display_decision_boundary(X, Y, classifier):
    h = 0.2
    # Set up the grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict the function value for the whole grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = classifier.fcann2_classify(grid_points)
    Z = Z.reshape(xx.shape)
    
    # Plot the contour and the points
    plt.contourf(xx, yy, Z, cmap=plt.cm.terrain, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral, edgecolor='k')
    
    # Draw the black line at the border between the classes
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)  
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.random.seed(10010)

    # Set hyperparameters
    param_niter = 5000
    param_delta = 0.05 
    param_lambda = 1e-3
    hidden_dim = 5

    # Generate data
    K = 6
    C = 2
    N = 10
    X, Y_ = sample_gmm_2d(K, C, N)

    # Initialize the model
    input_dim = X.shape[1]
    output_dim = C
    classifier = FCANN2(input_dim, hidden_dim, output_dim, param_delta, param_lambda)

    # Train the model
    classifier.fcann2_train(X, Y_, num_iterations=param_niter)

    # Display the decision boundary
    display_decision_boundary(X, Y_, classifier)
    
