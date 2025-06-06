# Suppress loky core detection warning
import os
import warnings
warnings.filterwarnings("ignore", message="Could not find the number of physical cores*")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Set based on your physical cores
# Import libraries
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import random

# Load the Iris dataset
iris = load_iris()
X = iris.data       # Features
y = iris.target     # Target labels
labels = iris.target_names

# Display random sample data
print("\nSample Data from Iris Dataset")
print("*" * 40)
for _ in range(10):
    i = random.randint(0, len(X) - 1)
    print(f"{X[i]}  ===>  {labels[y[i]]}")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


try:
    # Get K value from the user
    k = int(input("\nEnter the number of neighbors (K): "))
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Show model accuracy
    score = knn.score(X_test, y_test)
    print(f"\nModel Accuracy: {score:.2f}")

    # Get user input for prediction
    user_input = input("\nEnter 4 comma-separated values for prediction (e.g., 5.1,3.5,1.4,0.2):\n")
    test_data = [float(x.strip()) for x in user_input.split(",")]

    if len(test_data) != 4:
        raise ValueError("You must enter exactly 4 numeric values.")

    prediction = knn.predict([test_data])
    print("Predicted class:", labels[prediction[0]])

except ValueError as e:
    print("\nInvalid input:", e)
except Exception as e:
    print("\nAn error occurred:", e)
