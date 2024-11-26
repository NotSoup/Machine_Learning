from sklearn.model_selection import learning_curve, validation_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score
import numpy as np
import matplotlib.pyplot as plt


def BoostDT(X_train, X_test, y_train, y_test, ax, metric):

    ada_boost = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(
            random_state=90210,
            max_depth=3,
            min_samples_leaf=10,
        ),
        random_state=90210,
        n_estimators=10,
    )

    ###################### Learning Curve ######################
    # train_sizes, train_scores, valid_scores = learning_curve(
    #     ada_boost, 
    #     X_train, 
    #     y_train, 
    #     cv=5, 
    #     scoring=metric, 
    #     n_jobs=-1
    # )

    # train_mean = np.mean(train_scores, axis=1)
    # valid_mean = np.mean(valid_scores, axis=1)

    # plt.clf()
    # plt.figure(figsize=(6,6)) # or (7,6)
    # plt.plot(train_sizes, train_mean, label='Training Recall', color='blue', marker='o')
    # plt.plot(train_sizes, valid_mean, label='Validation Recall', color='red', marker='o')
    # plt.title('Learning Curve for AdaBoost Decision Tree')
    # plt.xlabel('Training Set Size')
    # plt.ylabel(metric)
    # plt.legend(loc='best')
    # plt.grid()
    # plt.savefig(f'./plots/o_dt_learn.png', dpi=300)

    ###################### Validation Curve ######################
    # base = 'base_estimator__'

    # param = 'max_depth'
    # param_range = [1, 2, 3, 5, 7, 10]       # 3   # 5
    # param = 'min_samples_leaf'
    # param_range = [1, 2, 5, 10, 20]             # 10    # 3
    # param = 'n_estimators'
    # param_range = [2, 5, 10, 20, 50]            # 10    # 20


    # train_scores, valid_scores = validation_curve(
    #     AdaBoostClassifier(
    #         base_estimator=DecisionTreeClassifier(
    #             random_state=90210,
    #             max_depth=5,
    #             min_samples_leaf=3,
    #         ), 
    #         # n_estimators=20,
    #         random_state=90210, 
    #     ),
    #     X_train, 
    #     y_train, 
    #     # param_name=base + param,
    #     param_name=param,
    #     param_range=param_range,
    #     cv=5,                                     
    #     scoring=metric,
    #     n_jobs=-1,
    # )

    # train_mean = np.mean(train_scores, axis=1)
    # valid_mean = np.mean(valid_scores, axis=1)

    # plt.clf()
    # plt.figure(figsize=(6,6)) # or (7,6)
    # plt.plot(param_range, train_mean, label='Training Recall', color='blue', marker='o')
    # plt.plot(param_range, valid_mean, label='Cross-Validation Recall', color='green', marker='o')
    # plt.title(f'Validation Curve for DT with Boost Classifier {param}')
    # plt.xlabel(f'{param}')
    # plt.ylabel(metric)
    # plt.legend(loc='best')
    # plt.grid()
    # plt.savefig(f'./plots/o_dt_{param}.png', dpi=300)

    # The test
    ada_boost.fit(X_train, y_train)
    y_pred = ada_boost.predict(X_test)

    if metric == 'accuracy':
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Boost-DR Test Accuracy: {accuracy:.2f}\n")
    if metric == 'recall_macro':
        recall = recall_score(y_test, y_pred, average='macro')
        print(f"DT Test Recall: {recall:.2f}\n")