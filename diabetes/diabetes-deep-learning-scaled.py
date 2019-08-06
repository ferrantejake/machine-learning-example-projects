from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

diabetes = pd.read_csv('diabetes.csv')

# Setup model
x_train, x_test, y_train, y_test = train_test_split(
    diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'], \
    stratify=diabetes['Outcome'], random_state=66)

# Scale Data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

## MLP implmentation
mlp = MLPClassifier(random_state=28, max_iter=160)
mlp.fit(x_train_scaled, y_train)

print("Accuracy on training set: {:.2f}".format(mlp.score(x_train_scaled, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score(x_test_scaled, y_test)))
