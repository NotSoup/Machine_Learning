from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np


# These are to ignore the warning: "ConvergenceWarning: Stochastic Optimizer: Maximum iterations (500) reached and the optimization hasn't converged yet."
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
@ignore_warnings(category=ConvergenceWarning)
def NN(X, y):
    
    #####
    # Reqired-to-be-tuned hyperparameters: hidden layer size (width, depth)
    #   +2 more
    #####

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=90210)

    # Create the neural network classifier
    # Here we use one hidden layer with 5 neurons; you can adjust this as needed
    nn = MLPClassifier(hidden_layer_sizes=(5,), max_iter=500, random_state=90210)

    # Perform cross-validation (5-fold)
    cv_scores = cross_val_score(nn, X_train, y_train, cv=4)

    # # Print the cross-validation scores for each fold
    # print(f"Cross-Validation Scores: {cv_scores}")

    # Print the average cross-validation score
    print(f"NN Avg CV Score:  {np.mean(cv_scores):.2f}")

    # Train the classifier
    nn.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = nn.predict(X_test)

    # Evaluate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"NN Test Accuracy: {accuracy:.2f}\n")

    return accuracy