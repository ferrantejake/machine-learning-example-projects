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
num_neighbors = 3
knn = KNeighborsClassifier(n_neighbors=num_neighbors)
knn.fit(x_train, y_train)

print('Accuracy of K-NN ({}) classifier on training set: {:.2f}'.format(num_neighbors, knn.score(x_train, y_train)))
print('Accuracy of K-NN ({}) classifier on test set: {:.2f}'.format(num_neighbors, knn.score(x_test, y_test)))

## We saw that checking 17 neighbors worked better..
num_neighbors = 17
knn = KNeighborsClassifier(n_neighbors=num_neighbors)
knn.fit(x_train, y_train)

print('Accuracy of K-NN ({}) classifier on training set: {:.2f}'.format(num_neighbors, knn.score(x_train, y_train)))
print('Accuracy of K-NN ({}) classifier on test set: {:.2f}'.format(num_neighbors, knn.score(x_test, y_test)))
