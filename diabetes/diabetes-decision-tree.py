from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

diabetes = pd.read_csv('diabetes.csv')

# Convert features and outcomes into nparrays
x_train, x_test, y_train, y_test = train_test_split( \
    diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'], \
    stratify=diabetes['Outcome'], random_state=66)

## Decision Tree implmentation
mean_train_accuracy_by_depth = np.array([])
mean_test_accuracy_by_depth = np.array([])

num_depths_tried = 10
for i in range(1, num_depths_tried + 1):
    sum_train = 0
    sum_test = 0
    num_test_iterations = 50
    for j in range(0, num_test_iterations):
        tree = DecisionTreeClassifier(random_state=40, max_depth=i)
        tree.fit(x_train, y_train)
        sum_train += tree.score(x_train, y_train)
        sum_test += tree.score(x_test, y_test)

    mean_train = (sum_train / num_test_iterations) * 100
    mean_test = (sum_test / num_test_iterations) * 100

    print('[Decision Tree (max_depth={})] Accuracy on training set: {:.3f}'.format(i, mean_train))
    print('[Decision Tree (max_depth={})] Accuracy on test set: {:.3f}'.format(i, mean_test))

    mean_train_accuracy_by_depth = np.append(mean_train_accuracy_by_depth, mean_train)
    mean_test_accuracy_by_depth = np.append(mean_test_accuracy_by_depth, mean_test)

# Plotting
diff = abs(mean_train_accuracy_by_depth - mean_test_accuracy_by_depth)
plt.plot(range(1,11), mean_train_accuracy_by_depth, label='training accuracy')
plt.plot(range(1,11), mean_test_accuracy_by_depth, label='test accuracy')
plt.plot(range(1,11), diff, label='delta (lower is better)')
plt.title('Accuracy of prediction by tree max_depth (and delta)')
plt.ylabel('Accuracy (%)')
plt.xlabel('max_depth')
plt.legend()
plt.show()
