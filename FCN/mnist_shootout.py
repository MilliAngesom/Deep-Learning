import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


dataset_root = '/home/milli'  
mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=False)
mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=False)

x_train, y_train = mnist_train.data, mnist_train.targets
x_test, y_test = mnist_test.data, mnist_test.targets
x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

N = x_train.shape[0]
D = x_train.shape[1] * x_train.shape[2]  # shape of x_train = (50000, 28, 28)
C = y_train.max().add_(1).item()

# Flatten the images
x_train = x_train.view(N, D)
x_test = x_test.view(-1, D)

# Define the PTDeep model class
class PTDeep(torch.nn.Module):
    def __init__(self, dims):
        super(PTDeep, self).__init__()
        self.dims = dims
        self.layers = torch.nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(torch.nn.Linear(dims[i], dims[i + 1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)  
        return x

    def predict(self, x):
        logits = self.forward(x)
        return logits.argmax(dim=1)



def train(model, x_train_full, y_train_full, x_test, y_test, epochs, lr, reg, patience=10):

    # Split the training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=1/5, random_state=42)

    # Define the loss function and the optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

    # Store the losses and performance metrics
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    precision_scores = []
    recall_scores = []
    top_loss_images = []
    all_top_loss_images = []
    all_top_loss_images_label = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    counter = 0  # Counter for early stopping

    # Loop over the epochs
    for epoch in range(epochs):
        # Training phase
        model.train()  # Set the model to training mode
        logits = model(x_train)
        loss = criterion(logits, y_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = criterion(val_logits, y_val)

        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            counter = 0  # Reset the counter if we found a new best model
        else:
            counter += 1  # Increment the counter if no improvement


        # Compute and store the losses
        train_loss = loss.item()
        test_loss = criterion(model(x_test), y_test).item()
        train_losses.append(train_loss)
        val_losses.append(val_loss.item())
        test_losses.append(test_loss)

        # Compute the train and test accuracies
        train_preds = model.predict(x_train)
        test_preds = model.predict(x_test)
        train_acc = accuracy_score(y_train.numpy(), train_preds.numpy())
        test_acc = accuracy_score(y_test.numpy(), test_preds.numpy())
        train_accs.append(train_acc) # store the train accuracy for plotting purpose
        test_accs.append(test_acc)   # store the test accuracy for plotting purpose


        # Compute precision and recall
        train_precision = precision_score(y_train.numpy(), train_preds.numpy(), average='macro')
        test_precision = precision_score(y_test.numpy(), test_preds.numpy(), average='macro')
        precision_scores.append((train_precision, test_precision))
        
        train_recall = recall_score(y_train.numpy(), train_preds.numpy(), average='macro')
        test_recall = recall_score(y_test.numpy(), test_preds.numpy(), average='macro')
        recall_scores.append((train_recall, test_recall))
        
        # Find images that contribute most to the loss
        loss_values = torch.nn.functional.cross_entropy(logits, y_train, reduction='none') # computes the loss for individual image and do not combine it to one value
        top_loss_indices = loss_values.topk(5, largest=True)[1]
        all_top_loss_images.append(x_train[top_loss_indices])
        all_top_loss_images_label.append(y_train[top_loss_indices])

        # Print the epoch summary
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss.item():.4f}, Test Loss: {test_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        
        print(f'Train Precision: {train_precision:.4f}, Test Precision: {test_precision:.4f}')
        print(f'Train Recall: {train_recall:.4f}, Test Recall: {test_recall:.4f}')


        # Early stopping check
        if counter >= patience:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break

    # Load the best model state
    model.load_state_dict(best_model_state)

    # Evaluate on the test set to find the top 5 loss-contributing images
    model.eval()  # Set the model to evaluation mode
    test_losses_ = []
    test_images = []
    with torch.no_grad():
        for i in range(len(x_test)):
            # Get the test image and label
            test_image = x_test[i].unsqueeze(0)  # To make sure the shape of x_test is torch.Size([1, 784]) and not [784]
            test_label = y_test[i].unsqueeze(0)

            # Compute the loss for this test image
            test_output = model(test_image)
            test_loss = criterion(test_output, test_label).item()

            # Store the loss and the image
            test_losses_.append(test_loss)
            test_images.append(test_image)

    # Find the top 5 images with the highest loss
    top_5_loss_indices = np.argsort(test_losses_)[-5:]
    top_5_images = [test_images[i] for i in top_5_loss_indices]
    top_5_losses = [test_losses_[i] for i in top_5_loss_indices]

    # Visualize the top 5 loss test images
    plt.figure(figsize=(15, 3))
    for idx, (image, loss) in enumerate(zip(top_5_images, top_5_losses)):
        plt.subplot(1, 5, idx + 1)
        # Reshape the image data from (784,) to (28, 28)
        image_2d = image.squeeze().cpu().numpy().reshape(28, 28)
        plt.imshow(image_2d, cmap='gray')
        plt.title(f'Loss: {loss:.2f}')
        plt.axis('off')
    plt.suptitle('Wrongly Classified Images by the Most Successful Model')
    plt.tight_layout()
    plt.show()


    # Find the top 5 loss training images across all epochs
    all_top_loss_images = torch.cat(all_top_loss_images, dim=0) # convert it to one larger tensor
    all_top_loss_images_label = torch.cat(all_top_loss_images_label, dim=0) # convert it to one larger tensor
    # loss_values = torch.nn.functional.cross_entropy(model(all_top_loss_images), y_train[:len(all_top_loss_images)], reduction='none')
    loss_values = torch.nn.functional.cross_entropy(model(all_top_loss_images), all_top_loss_images_label, reduction='none') 
    top_loss_indices = loss_values.topk(5, largest=True)[1]
    top_loss_images = all_top_loss_images[top_loss_indices]

    # Visualize the top loss images accumulated over all epochs
    plt.figure(figsize=(12, 6))
    for i, top_loss_image in enumerate(top_loss_images):
        plt.subplot(1, len(top_loss_images), i + 1)
        image = top_loss_image.view(28, 28).detach().numpy()
        plt.imshow(image, cmap='gray')
        plt.title(f'Top Loss Image {i+1}')
    plt.suptitle('Top Loss Images Accumulated Over All Epochs')
    plt.tight_layout()
    plt.show()

    # Plot the losses and accuracies
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Loss vs Epoch for PTDeep with dims {model.dims} and reg {reg}')
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(test_accs, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Accuracy vs Epoch for PTDeep with dims {model.dims} and reg {reg}')
    plt.tight_layout()
    plt.show()

    
    # Plot and comment on the weight matrices for each digit separately
    plt.figure(figsize=(10, 10))    
    for i in range(10):
        plt.subplot(4, 3, i+1)
        plt.imshow(model.layers[0].weight[i].view(28, 28).detach().numpy(), cmap='gray')
        plt.title(f'Weight Matrix for Digit {i}')
    plt.suptitle(f'Weight Matrices for PTDeep with dims {model.dims} and reg {reg}')
    plt.tight_layout()
    plt.show()
    
    # Comment on the weight matrices
    print(f'The weight matrices for PTDeep with dims {model.dims} and reg {reg} show the features that '
      'the model has learned to recognize each digit. The darker regions indicate the negative weights '
      'and the lighter regions indicate the positive weights. The weight matrices can be interpreted as '
      'the ideal images that the model expects to see for each digit. For example, the weight matrix for '
      'digit 0 has a lighter circle and a darker background, which matches the shape of 0. The weight matrix '
      'for digit 1 has a lighter vertical line and a darker background, which matches the shape of 1. The '
      'weight matrices for digits 2, 3, 4, 5, 6, 7, 8, and 9 also show similar patterns that correspond '
      'to their shapes.')

    # Return the model and loss histories
    return model, train_losses, val_losses




def train_mb(model, x_train_full, y_train_full, x_test, y_test, epochs, lr, reg, batch_size, patience=10):
    # Split the training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=1/5, random_state=42)

    # Define the loss function and the optimizer
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=reg)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    
    # Define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1-1e-4)


    # Store the losses and performance metrics
    train_losses = []
    val_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    precision_scores = []
    recall_scores = []
    all_top_loss_images = []
    all_top_loss_images_label = []
    best_val_loss = float('inf')
    best_model_state = None
    counter = 0  # Counter for early stopping

    # Calculate the number of batches
    num_batches = int(np.ceil(x_train.shape[0] / batch_size))

    # Loop over the epochs
    for epoch in range(epochs):
        # Shuffle the training data at the beginning of each epoch
        permutation = torch.randperm(x_train.size()[0])
        x_train_shuffled = x_train[permutation]
        y_train_shuffled = y_train[permutation]

        # Training phase
        model.train()  # Set the model to training mode
        for batch in range(num_batches):
            # Select the mini-batch
            start = batch * batch_size
            end = min(start + batch_size, x_train.size()[0])
            x_batch = x_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            # Forward pass
            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Store the loss
            train_losses.append(loss.item())

            # Store top loss images for visualization
            loss_values = torch.nn.functional.cross_entropy(logits, y_batch, reduction='none')
            top_loss_indices = loss_values.topk(5, largest=True)[1]
            all_top_loss_images.append(x_batch[top_loss_indices])
            all_top_loss_images_label.append(y_batch[top_loss_indices])
        
        

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = criterion(val_logits, y_val)

        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            counter = 0  # Reset the counter if we found a new best model
        else:
            counter += 1  # Increment the counter if no improvement

        # Update the learning rate
        scheduler.step()

        # Compute and store the validation loss
        val_losses.append(val_loss.item())

        # Compute the train and test accuracies
        train_acc = accuracy_score(y_train.numpy(), model.predict(x_train).numpy())
        test_acc = accuracy_score(y_test.numpy(), model.predict(x_test).numpy())
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        # Compute precision and recall
        train_precision = precision_score(y_train.numpy(), model.predict(x_train).numpy(), average='macro')
        test_precision = precision_score(y_test.numpy(), model.predict(x_test).numpy(), average='macro')
        precision_scores.append((train_precision, test_precision))
        
        train_recall = recall_score(y_train.numpy(), model.predict(x_train).numpy(), average='macro')
        test_recall = recall_score(y_test.numpy(), model.predict(x_test).numpy(), average='macro')
        recall_scores.append((train_recall, test_recall))

        # Early stopping check
        if counter >= patience:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break

        # Print the epoch summary
        print(f'Epoch {epoch + 1}, Train Loss: {np.mean(train_losses[-num_batches:]):.4f}, '
              f'Validation Loss: {val_loss.item():.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        print(f'Train Precision: {train_precision:.4f}, Test Precision: {test_precision:.4f}')
        print(f'Train Recall: {train_recall:.4f}, Test Recall: {test_recall:.4f}')

    # Load the best model state
    model.load_state_dict(best_model_state)

    # Evaluate on the test set to find the top 5 loss-contributing images
    model.eval()  # Set the model to evaluation mode
    test_images = []
    with torch.no_grad():
        for i in range(len(x_test)):
            # Get the test image and label
            test_image = x_test[i].unsqueeze(0)  
            test_label = y_test[i].unsqueeze(0)

            # Compute the loss for this test image
            test_output = model(test_image)
            test_loss = criterion(test_output, test_label).item()

            # Store the loss and the image
            test_losses.append(test_loss)
            test_images.append(test_image)

    # Find the top 5 images with the highest loss
    top_5_loss_indices = np.argsort(test_losses)[-5:]
    top_5_images = [test_images[i] for i in top_5_loss_indices]
    top_5_losses = [test_losses[i] for i in top_5_loss_indices]

    # Visualize the top 5 loss test images
    plt.figure(figsize=(15, 3))
    for idx, (image, loss) in enumerate(zip(top_5_images, top_5_losses)):
        plt.subplot(1, 5, idx + 1)
        # Reshape the image data from (784,) to (28, 28)
        image_2d = image.squeeze().cpu().numpy().reshape(28, 28)
        plt.imshow(image_2d, cmap='gray')
        plt.title(f'Loss: {loss:.2f}')
        plt.axis('off')
    plt.suptitle('Wrongly Classified Images by the Most Successful Model')
    plt.tight_layout()
    plt.show()


    # Visualize the top loss train images accumulated over all epochs
    all_top_loss_images = torch.cat(all_top_loss_images, dim=0) # convert it to one larger tensor
    all_top_loss_images_label = torch.cat(all_top_loss_images_label, dim=0) # convert it to one larger tensor
    loss_values = torch.nn.functional.cross_entropy(model(all_top_loss_images), all_top_loss_images_label, reduction='none') 
    top_loss_indices = loss_values.topk(5, largest=True)[1]
    top_loss_images = all_top_loss_images[top_loss_indices]

    plt.figure(figsize=(12, 6))
    for i, top_loss_image in enumerate(top_loss_images):
        plt.subplot(1, 5, i + 1)
        image = np.reshape(top_loss_image, (28, 28))  
        plt.imshow(image, cmap='gray')
        plt.title(f'Top Loss Image {i+1}')
    plt.suptitle('Top Loss Images Accumulated Over All Epochs')
    plt.tight_layout()
    plt.show()

    # Plot the losses and accuracies
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Loss vs Epoch for PTDeep with dims {model.dims} and reg {reg}')
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(test_accs, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Accuracy vs Epoch for PTDeep with dims {model.dims} and reg {reg}')
    plt.tight_layout()
    plt.show()

    
    # Plot and comment on the weight matrices for each digit separately
    plt.figure(figsize=(10, 10))    
    for i in range(10):
        plt.subplot(4, 3, i+1)
        plt.imshow(model.layers[0].weight[i].view(28, 28).detach().numpy(), cmap='gray')
        plt.title(f'Weight Matrix for Digit {i}')
    plt.suptitle(f'Weight Matrices for PTDeep with dims {model.dims} and reg {reg}')
    plt.tight_layout()
    plt.show()
    
    # Comment on the weight matrices
    print(f'The weight matrices for PTDeep with dims {model.dims} and reg {reg} show the features that '
      'the model has learned to recognize each digit. The lighter regions indicate the positive weights '
      'and the darker regions indicate the negative weights. The weight matrices can be interpreted as '
      'the ideal images that the model expects to see for each digit. For example, the weight matrix for '
      'digit 0 has a lighter circle and a darker background, which matches the shape of 0. The weight matrix '
      'for digit 1 has a lighter vertical line and a darker background, which matches the shape of 1. The '
      'weight matrices for digits 2, 3, 4, 5, 6, 7, 8, and 9 also show similar patterns that correspond '
      'to their shapes.')

    # Return the model and loss histories
    return model, train_losses, val_losses, train_accs, test_accs, precision_scores, recall_scores

def train_svm_classifier(x_train, y_train, x_test, y_test, kernel_type='linear'):
    """
    Trains an SVM classifier with the specified kernel type and evaluates its accuracy on the test set.
    
    Parameters:
    x_train (array-like): Training data features.
    y_train (array-like): Training data labels.
    x_test (array-like): Test data features.
    y_test (array-like): Test data labels.
    kernel_type (str): Type of SVM kernel ('linear' or 'rbf').
    
    Returns:
    float: Accuracy of the SVM classifier on the test set.
    """
    # Create a pipeline that standardizes the data and then applies the SVM classifier
    svm_classifier = make_pipeline(StandardScaler(), SVC(kernel=kernel_type, decision_function_shape='ovo'))
    
    # Train the SVM classifier
    svm_classifier.fit(x_train, y_train)
    
    # Predict the labels for the test set
    predictions = svm_classifier.predict(x_test)
    
    # Calculate and return the accuracy
    accuracy = accuracy_score(y_test, predictions)
    return accuracy


# Define the main function
def main():
    # Define the dimensions for the PTDeep model
    dims = [D, C]  # Input layer (D), Hidden layers (100, 100), Output layer (C)
    
    # Create the model
    model = PTDeep(dims)
    
    # Train the model with different regularization values
    regs = [0.000]
    for reg in regs:
        print(f'Training the model with regularization {reg}')
        train(model, x_train, y_train, x_test, y_test, epochs=100, lr=0.001, reg=reg, patience=10)
        # train_mb(model, x_train, y_train, x_test, y_test, epochs=100, lr=0.0001, reg=reg, batch_size=10)

    # # Train and explain the loss function of a randomly initialized model (that has not seen any training data)
    # # Define the loss function
    # criterion = torch.nn.CrossEntropyLoss()
    # # Evaluate the model on the test data without training
    # model.eval()  # Set the model to evaluation mode
    # with torch.no_grad():
    #     # Forward pass through the model
    #     logits = model(x_test)
    #     # Calculate the loss
    #     loss = criterion(logits, y_test)
    #     # Calculate the predictions
    #     _, predicted = torch.max(logits.data, 1)
    #     # Calculate accuracy
    #     correct = (predicted == y_test).sum().item()
    #     accuracy = correct / y_test.size(0)
    # # Print the initial loss
    # print(f'Initial loss of the randomly initialized model: {loss.item()}')
    # print(f'Initial accuracy of the randomly initialized model: {accuracy:.4f}')



    # # Train and evaluate a linear SVM classifier
    # linear_svm_accuracy = train_svm_classifier(x_train, y_train, x_test, y_test, kernel_type='linear')
    # print(f"Linear SVM accuracy: {linear_svm_accuracy:.4f}")

    # # Train and evaluate an RBF kernel SVM classifier
    # rbf_svm_accuracy = train_svm_classifier(x_train, y_train, x_test, y_test, kernel_type='rbf')
    # print(f"RBF kernel SVM accuracy: {rbf_svm_accuracy:.4f}")

# Run the main function when the file is called
if __name__ == "__main__":
    main()

