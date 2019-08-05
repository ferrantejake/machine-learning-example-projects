from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load the iris dataset
iris = load_iris()

x = iris.data  # Array of the data
# Array of labels of each data entry, there are 3 here for the different classes
y = iris.target

# Take random indicies to split the dataset into train and test
test_ids = np.random.permutation(len(x))

# Split data and labels into train and test
# Keep the last 10 entries for testing

x_train = x[test_ids[:-10]]
x_test = x[test_ids[-10:]]

y_train = y[test_ids[:-10]]
y_test = y[test_ids[-10:]]

# Classify using decision tree
tree = DecisionTreeClassifier()

# Tain the classifier with the training set
tree.fit(x_train, y_train)

# Predict on the test dataset
pred = tree.predict(x_test)

# Output
print pred
print y_test
print (accuracy_score(pred, y_test)) * 100
