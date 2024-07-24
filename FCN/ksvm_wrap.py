import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, precision_score, average_precision_score
from DATA import *


def graph_data(X, Y_, Y, special=[]):
  """Creates a scatter plot (visualize with plt.show)

  Arguments:
      X:       datapoints
      Y_:      groundtruth classification indices
      Y:       predicted class indices
      special: use this to emphasize some points

  Returns:
      None
  """
  # colors of the datapoint markers
  palette=([0.5,0.5,0.5], [1,1,1], [0.2,0.2,0.2])
  colors = np.tile([0.0,0.0,0.0], (Y_.shape[0],1))
  for i in range(len(palette)):
    colors[Y_==i] = palette[i]

  # sizes of the datapoint markers
  sizes = np.repeat(20, len(Y_))
  sizes[special] = 40
  
  # draw the correctly classified datapoints
  good = (Y_==Y)
  plt.scatter(X[good,0],X[good,1], c=colors[good], 
              s=sizes[good], marker='o', edgecolors='black')

  # draw the incorrectly classified datapoints
  bad = (Y_!=Y)
  plt.scatter(X[bad,0],X[bad,1], c=colors[bad], 
              s=sizes[bad], marker='s', edgecolors='black')
  # Emphasize the support vectors
  plt.scatter(X[special,0], X[special,1], c='none', edgecolors='red', 
              s=sizes[special], marker='o', linewidths=2)


class KSVMWrap:
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.clf = svm.SVC(C=param_svm_c, gamma=param_svm_gamma)
        self.clf.fit(X, Y_)

    def predict(self, X):
        return self.clf.predict(X)

    def get_scores(self, X):
        return self.clf.decision_function(X)

    @property
    def support(self):
        return self.clf.support_


# Generate some data
np.random.seed(100)
K = 6
C = 2
N = 10
param_svm_c = 1  # A high value of C prioritizes minimizing training errors and minimizes the margin
param_svm_gamma = 'auto'  # high value of gamma gives more weight to the support vectors 
X, Y_ = sample_gmm_2d(K, C, N)

# Train the RBF SVM classifier
ksvm = KSVMWrap(X, Y_, param_svm_c=param_svm_c, param_svm_gamma=param_svm_gamma)

# Define a grid of points for the decision surface
h = 0.01
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Get decision surface predictions
Z = ksvm.predict(grid_points)
Z = Z.reshape(xx.shape)

# Get support vectors
support = ksvm.support


# Make predictions on the original data
Y_pred = ksvm.predict(X)

# Calculate accuracy, recall, and precision
accuracy = accuracy_score(Y_, Y_pred)
recall = recall_score(Y_, Y_pred, average='macro')
precision = precision_score(Y_, Y_pred, average='macro')

# For average precision, you'll need classification scores
scores = ksvm.get_scores(X)
avg_precision = average_precision_score(Y_, scores)

# Display the performance metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Average Precision: {avg_precision:.4f}")

# Visualize the data and decision surface with class color information
palette = plt.cm.get_cmap("tab20", C)  # Use a color map for 'C' classes
colors = palette(Y_)

plt.contourf(xx, yy, Z, cmap=palette, alpha=0.8)
graph_data(X, Y_, ksvm.predict(X), special=support)
plt.scatter(X[:, 0], X[:, 1], c=colors, cmap=palette, marker='o', edgecolors='black')
plt.title("RBF SVM Decision Surface")
plt.show()
