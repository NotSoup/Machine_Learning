from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


def BoostDT(X, y):

    #####
    # Reqired-to-be-tuned hyperparameters: {Boosting > # of weak learners} {DT > Pruning}
    #   +2 more
    #####

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=90210)

    # Initialize the Gradient Boosting Classifier
    gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=90210)
    
    # Perform cross-validation (5-fold)
    cv_scores = cross_val_score(gb_clf, X_train, y_train, cv=4)

    # # Print the cross-validation scores for each fold
    # print(f"Cross-Validation Scores: {cv_scores}")

    # Print the average cross-validation score
    print(f"Boost-DR Avg CV Score:  {np.mean(cv_scores):.2f}")

    # Fit the model to the training data
    gb_clf.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = gb_clf.predict(X_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Boost-DR Test Accuracy: {accuracy:.2f}\n")

    # Learning Curve: Model performance with varying training set sizes
    train_sizes, train_scores, valid_scores = learning_curve(gb_clf, X_train, y_train, cv=4, scoring='accuracy', n_jobs=-1)

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

    # # Optional: Visualize the importance of each feature
    # feature_importances = gb_clf.feature_importances_
    # indices = np.argsort(feature_importances)[::-1]

    # plt.figure(figsize=(10, 6))
    # plt.title("Feature Importances")
    # plt.bar(range(X.shape[1]), feature_importances[indices], align="center")
    # plt.xticks(range(X.shape[1]), indices)
    # plt.xlabel("Feature Index")
    # plt.ylabel("Importance")
    # plt.show()
