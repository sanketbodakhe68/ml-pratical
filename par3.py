import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the diabetes dataset
diabetes = load_diabetes()

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)

# The target variable
target = diabetes.target

# For simplicity, let's use only one feature (e.g., 'BMI') for simple linear regression
X = df[['bmi']].values  # Independent variable
y = target  # Dependent variable

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Initialize the Linear Regression model
model = LinearRegression()

# Step 4: Train the model
model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Step 7: Visualize the results
plt.scatter(X_test, y_test, color='blue', label='True values')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression line')
plt.title('Simple Linear Regression - BMI vs Diabetes Progression')
plt.xlabel('BMI')
plt.ylabel('Diabetes Progression')
plt.legend()
plt.show()
