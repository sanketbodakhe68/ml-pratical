import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
# Load data
# Download the data in csv file 
# set the Path of your download the data in csv file 
data = pd.read_csv(r'partical 5\tennis.csv')

print("The first 5 values of data is:\n", data.head())
# Splitting features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
#print the first five value of the data set. this use to the above imorted csv file 
print("\nThe First 5 values of train data is\n", X.head())
print("\nThe first 5 values of Train output is\n", y.head())
# Encoding categorical variables
le_Outlook = LabelEncoder()
X['Outlook'] = le_Outlook.fit_transform(X['Outlook'])
# Encoding categorical variables
le_Temperature = LabelEncoder()
X['Temperature'] = le_Temperature.fit_transform(X['Temperature'])
# Encoding categorical variables
le_Humidity = LabelEncoder()
X['Humidity'] = le_Humidity.fit_transform(X['Humidity'])
# Encoding categorical variables
le_Windy = LabelEncoder()
X['Windy'] = le_Windy.fit_transform(X['Windy'])
# print the train data 
print("\nNow the Train data is:\n", X.head())
le_PlayTennis = LabelEncoder()
y = le_PlayTennis.fit_transform(y)
print("\nNow the Train output is:\n", y)
# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# Train Naive Bayes classifier
classifier = GaussianNB()
classifier.fit(X_train, y_train)
# Predict and evaluate
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)
print("Accuracy is:", accuracy)
# extara part of the code to plot the confusion matarix in the output screen in use of plot 
#evalaute the confusion matrix 
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
# Plot confusion matrix using seaborn heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le_PlayTennis.classes_,
yticklabels=le_PlayTennis.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()