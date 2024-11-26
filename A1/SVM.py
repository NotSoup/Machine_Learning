from sklearn.model_selection import learning_curve, validation_curve
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score
import numpy as np
import matplotlib.pyplot as plt


def SVM(X_train, X_test, y_train, y_test, ax, metric):

    svm = SVC(
        random_state=90210,
        kernel='linear',    # poly
        C=20,              #1e6
                            # degree=3
    ) 

    ###################### Learning Curve ######################
    # train_sizes, train_scores, valid_scores = learning_curve(
    #     svm, 
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
    # plt.plot(train_sizes, train_mean, label='Training Accuracy', color='blue', marker='o')
    # plt.plot(train_sizes, valid_mean, label='Validation Accuracy', color='red', marker='o')
    # plt.title('Learning Curve for SVM')
    # plt.xlabel('Training Set Size')
    # plt.ylabel(metric)
    # plt.legend(loc='best')
    # plt.grid()
    # plt.savefig(f'./plots/r_svm_learn.png', dpi=300)

    ###################### Validation Curve ######################
    # param_range = ['linear', 'poly', 'rbf', 'sigmoid'] # 'linear'     poly
    # param = 'kernel'
    # param_range = [5, 7, 9, 10, 11, 15, 20, 50, 100]         # 100      20
    # param = 'C'
    # param_range = [2, 3, 4, 5]
    # param = 'degree'

    # train_scores, valid_scores = validation_curve(
    #     svm,
    #     X_train, 
    #     y_train,                      
    #     param_name=param,
    #     param_range=param_range,
    #     cv=3,
    #     scoring=metric,
    #     n_jobs=-1,
    # )

    # train_mean = np.mean(train_scores, axis=1)
    # valid_mean = np.mean(valid_scores, axis=1)

    # plt.clf()
    # plt.figure(figsize=(6,6)) # or (7,6)
    # # plt.xscale('log')
    
    # # # Kernel
    # # plt.bar(param_range, train_mean, label='Training Accuracy', color='blue')
    # # plt.bar(param_range, valid_mean, label='Validation Accuracy', color='green')
    # # plt.title('Kernel for SVM')
    
    # plt.plot(param_range, train_mean, label='Training Accuracy', color='blue', marker='o')
    # plt.plot(param_range, valid_mean, label='Cross-Validation Accuracy', color='green', marker='o')
    # plt.title(f'Validation Curve for SVM {param}')
    # plt.xlabel(f'{param}')
    # plt.ylabel(metric)
    # plt.legend(loc='best')
    # plt.grid()
    # plt.savefig(f'./plots/r_svm_{param}.png', dpi=300)

    # The test
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    if metric == 'accuracy':
        accuracy = accuracy_score(y_test, y_pred)
        print(f"SVM Test Accuracy: {accuracy:.2f}\n")
    if metric == 'recall_macro':
        recall = recall_score(y_test, y_pred, average='macro')
        print(f"SVM Test Recall: {recall:.2f}\n")