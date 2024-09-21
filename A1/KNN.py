from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


def KNN(X, y):

    #####
    # Reqired-to-be-tuned hyperparameters: K
    #   +2 more
    #####

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=90210)

    # Create the k-NN classifier
    k = 1  # Number of neighbors to use
    knn = KNeighborsClassifier(n_neighbors=k)

    # Perform cross-validation (5-fold)
    cv_scores = cross_val_score(knn, X_train, y_train, cv=3)

    # # Print the cross-validation scores for each fold
    # print(f"Cross-Validation Scores: {cv_scores}")

    # Print the average cross-validation score
    print(f"KNN Avg CV Score:  {np.mean(cv_scores):.2f}")

    # Train the classifier
    knn.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = knn.predict(X_test)

    # Evaluate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"KNN Test Accuracy: {accuracy:.2f}\n")

    # Learning Curve: Model performance with varying training set sizes
    train_sizes, train_scores, valid_scores = learning_curve(knn, X_train, y_train, cv=4, scoring='accuracy', n_jobs=-1)

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

    # # Validation Curve: Model performance with varying number of neighbors
    # param_range = np.arange(1, 21)  # Trying k values from 1 to 20
    # train_scores_vc, valid_scores_vc = validation_curve(knn, X_train, y_train, param_name='n_neighbors',
    #                                                     param_range=param_range, cv=5, scoring='accuracy', n_jobs=-1)

    # # Compute average training and validation scores for each k value
    # train_mean_vc = np.mean(train_scores_vc, axis=1)
    # valid_mean_vc = np.mean(valid_scores_vc, axis=1)

    # # Plotting the Validation Curve
    # plt.figure(figsize=(8, 6))
    # plt.plot(param_range, train_mean_vc, label='Training Accuracy', color='blue')
    # plt.plot(param_range, valid_mean_vc, label='Validation Accuracy', color='red')
    # plt.title('Validation Curve for k-NN')
    # plt.xlabel('Number of Neighbors (k)')
    # plt.ylabel('Accuracy')
    # plt.legend(loc='best')
    # plt.grid()
    # plt.show()


    return accuracy