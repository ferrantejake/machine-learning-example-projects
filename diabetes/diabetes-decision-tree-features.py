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
tree = DecisionTreeClassifier(random_state=40, max_depth=3)
tree.fit(x_train, y_train)

def plot_feature_tree_importances_diabetes(features, model):
    plt.figure(figsize=(len(features),6))
    n_features=len(features)
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), features)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.ylim(-1, n_features)
    plt.show()

features = [x for i,x in enumerate(diabetes.columns) if i!=8]
plot_feature_tree_importances_diabetes(features, tree)
