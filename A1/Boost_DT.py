from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np


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

    # # Print classification report
    # print("\nClassification Report:")
    # print(classification_report(y_test, y_pred))

    return accuracy

    # # Optional: Visualize the importance of each feature
    # import matplotlib.pyplot as plt

    # feature_importances = gb_clf.feature_importances_
    # indices = np.argsort(feature_importances)[::-1]

    # plt.figure(figsize=(10, 6))
    # plt.title("Feature Importances")
    # plt.bar(range(X.shape[1]), feature_importances[indices], align="center")
    # plt.xticks(range(X.shape[1]), indices)
    # plt.xlabel("Feature Index")
    # plt.ylabel("Importance")
    # plt.show()
