from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

diabetes = pd.read_csv('diabetes.csv')

# Setup model
x_train, x_test, y_train, y_test = train_test_split(
    diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'], \
    stratify=diabetes['Outcome'], random_state=66)

## MLP implmentation
mlp = MLPClassifier(random_state=8)
mlp.fit(x_train, y_train)

print("Accuracy on training set: {:.2f}".format(mlp.score(x_train, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score(x_test, y_test)))
