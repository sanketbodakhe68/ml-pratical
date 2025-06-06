#import the nessory labrieries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load dataset
# You can download this CSV from: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv

# Column names for the dataset
column_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

# Load data from CSV
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
df = pd.read_csv(url, names=column_names)

# Display sample data
print("\nSample Data:")
print(df.head())

# Split features and labels
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Initialize and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Predict with user input
try:
    print("\nEnter patient data for prediction:")
    input_values = input("Enter 8 values separated by commas (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age):\n")
    user_data = [float(i) for i in input_values.split(",")]

    if len(user_data) != 8:
        raise ValueError("Exactly 8 numeric values are required.")

    prediction = model.predict([user_data])
    print("\nPrediction: Diabetic" if prediction[0] == 1 else "\nPrediction: Not Diabetic")

except Exception as e:
    print("Error:", e)


# Assume the model was trained with a DataFrame with columns ['feature1', 'feature2']
