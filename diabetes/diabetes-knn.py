from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

diabetes = pd.read_csv('diabetes.csv')

# Setup model
x_train, x_test, y_train, y_test = train_test_split( \
    diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'], \
    stratify=diabetes['Outcome'], random_state=66)

## KNN implmentation
training_accuracy = []
test_accuracy = []
neighbor_settings = range(1,21)

for n_neighbors in neighbor_settings:
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)
    # record training set accuracy
    score = knn.score(x_train, y_train)
    training_accuracy.append(score)
    # record test set accuracy
    score = knn.score(x_test, y_test)
    test_accuracy.append(score)

# Plotting
plt.plot(neighbor_settings, training_accuracy, label="training accuracy")
plt.plot(neighbor_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
# plt.savefig('knn_compare_model')
plt.show()
