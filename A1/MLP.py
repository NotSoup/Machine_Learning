from sklearn.model_selection import train_test_split, cross_val_score,learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


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

    # Learning Curve: Model performance with varying training set sizes
    train_sizes, train_scores, valid_scores = learning_curve(nn, X_train, y_train, cv=4, scoring='accuracy', n_jobs=-1)

    # Compute average training and validation scores across the n-folds
    train_mean = np.mean(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)

    # Plotting the Learning Curve
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, label='Training Accuracy', color='blue', marker='o')
    plt.plot(train_sizes, valid_mean, label='Validation Accuracy', color='red', marker='o')
    plt.title('Learning Curve')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

    return accuracy