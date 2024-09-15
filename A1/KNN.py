from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def KNN(X, y):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create the k-NN classifier
    k = 5  # Number of neighbors to use
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the classifier
    knn.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = knn.predict(X_test)

    # Evaluate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"KNN Accuracy: {accuracy:.2f}")

    return accuracy
    # Example of predicting a new sample
    # sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example sample
    # prediction = knn.predict(sample)
    # print(f"Predicted class for the sample: {iris.target_names[prediction][0]}")