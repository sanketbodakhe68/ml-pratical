#import libires function 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
#add your database file your pc in csv file 
df = pd.read_csv("partical 4\Real estate.csv")
df.drop('No', inplace=True, axis=1)  # Dropping unnecessary column
print(df.head())  # Display the first few rows of the dataset
print(df.columns)  # Display column names
# Plotting a scatterplot
sns.scatterplot(
    x='X4 number of convenience stores',
    y='Y house price of unit area',
    data=df
)
plt.title('Convenience Stores vs House Price')
plt.xlabel('Number of Convenience Stores')
plt.ylabel('House Price per Unit Area')
plt.show()
# Creating feature and target variables
X = df.drop('Y house price of unit area', axis=1)  # Feature variables
y = df['Y house price of unit area']  # Target variable
print(X.head())  # Display the first few rows of features
print(y.head())  # Display the first few rows of the target variable

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101
)
# Creating and fitting the regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Making predictions
predictions = model.predict(X_test)
# Model evaluation
print('Mean Squared Error:', mean_squared_error(y_test, predictions))
print('Mean Absolute Error:', mean_absolute_error(y_test, predictions))