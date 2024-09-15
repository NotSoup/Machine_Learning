from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


def NN(X, y):
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
    print(f"NN Accuracy: {accuracy:.2f}")

    return accuracy