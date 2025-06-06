from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # for visualization
from sklearn.tree import plot_tree
# Preparing Data (Example using the Iris dataset)

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#Creating a Decision Tree Model and Initialize the model
dtc = DecisionTreeClassifier(random_state=42) 
#Creating a Decision Tree Model
#  Creating a Decision Tree Model
# step of decision tree model 
#Step 1: Identify the problem
#Step 2: Begin to structure the decision tree
#Step 3: Identify decision alternatives
#Step 4: Estimate payoffs or costs
#Step 5: Assign probabilities
#Step 6: Determine the potential outcomes
#Step 7: Analyze and select the best decision
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3]
}

grid_search = GridSearchCV(dtc, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
print("Best hyperparameters:", best_params)
#Making Predictions and Evaluating the Model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
#Visualizing the Decision Tree (Optional
plt.figure(figsize=(15, 10))
plot_tree(best_model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
plt.title("Decision Tree")
plt.show()