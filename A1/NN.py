import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# # Load the Iris dataset
# iris = load_iris(as_frame=True)
# X = iris.data
# y = iris.target

dt = pd.read_csv("/home/apetrolino3/Desktop/GaTech/Machine_Learning/data/student_dropout_success.csv", sep=";")
X = dt.drop(["Target"], axis=1)
y = dt["Target"]
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the neural network classifier
# Here we use one hidden layer with 5 neurons; you can adjust this as needed
nn = MLPClassifier(hidden_layer_sizes=(5,), max_iter=500, random_state=42)

# Train the classifier
nn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nn.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print the classification report
print("Classification Report:")
# print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Example of predicting a new sample
# sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example sample
# prediction = nn.predict(sample)
# print(f"Predicted class for the sample: {iris.target_names[prediction][0]}")
