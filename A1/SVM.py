from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np


def SVM(X, y):

    #####
    # Reqired-to-be-tuned hyperparameters: kernel type
    #   +2 more
    #####

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=90210)

    # Create the SVM classifier
    svm = SVC(kernel='linear', random_state=90210)  # You can change the kernel type: 'linear', 'poly', 'rbf', etc.

    # Perform cross-validation (5-fold)
    cv_scores = cross_val_score(svm, X_train, y_train, cv=4)

    # # Print the cross-validation scores for each fold
    # print(f"Cross-Validation Scores: {cv_scores}")

    # Print the average cross-validation score
    print(f"SVM Avg CV Score:  {np.mean(cv_scores):.2f}")

    # Train the classifier
    svm.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = svm.predict(X_test)

    # Evaluate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM Test Accuracy: {accuracy:.2f}\n")

    return accuracy