from sklearn.model_selection import learning_curve, validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score
import numpy as np
import matplotlib.pyplot as plt


def KNN(X_train, X_test, y_train, y_test, ax, metric):

    knn = KNeighborsClassifier(
        n_neighbors=50,
        metric='canberra',
        weights='distance'
    )

    ###################### Learning Curve ######################
    train_sizes, train_scores, valid_scores = learning_curve(
        knn, 
        X_train, 
        y_train, 
        cv=5, 
        scoring=metric, 
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)

    plt.clf()
    plt.figure(figsize=(6,6)) # or (7,6)
    plt.plot(train_sizes, train_mean, label='Training Accuracy', color='blue', marker='o')
    plt.plot(train_sizes, valid_mean, label='Validation Accuracy', color='red', marker='o')
    plt.title('Learning Curve for k-NN')
    plt.xlabel('Training Set Size')
    plt.ylabel(metric)
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('./plots/r_knn_learn.png', dpi=300)

    ###################### Validation Curve ######################
    # # param = 'n_neighbors'
    # # param_range = np.arange(1, 200, 1)                                                              # 10        # 50
    # # param = 'metric'
    # # param_range = ['braycurtis','canberra','chebyshev',
    # #         'correlation','cosine','euclidean', 'manhattan','minkowski', 'sqeuclidean']             # manhattan # canberra
    # param = 'weights'
    # param_range = ['uniform', 'distance']                                                           # distance  # distance

    # train_scores, valid_scores = validation_curve(
    #     knn, 
    #     X_train, 
    #     y_train, 
    #     param_name=param,                                              
    #     param_range=param_range, 
    #     cv=5, 
    #     scoring=metric, 
    #     n_jobs=-1
    # )

    # train_mean = np.mean(train_scores, axis=1)
    # valid_mean = np.mean(valid_scores, axis=1)

    # # Weighting
    # print(train_mean)
    # print(valid_mean)

    # plt.clf()
    # plt.figure(figsize=(6,6)) # or (7,6)
    # plt.plot(param_range, train_mean, label='Training Accuracy', color='blue')
    # plt.plot(param_range, valid_mean, label='Validation Accuracy', color='green')

    # # # For distance metric
    # # plt.figure(figsize=(9,9))
    # # plt.bar(param_range, train_mean, label='Training Accuracy', color='blue')
    # # plt.bar(param_range, valid_mean, label='Validation Accuracy', color='green')
    # # plt.xticks(rotation=45, ha='right')
    # plt.ylim([0.86, 0.93])
    # # plt.xlabel('Distance Metrics')
    # # # plt.title('Weighting Metrics for k-NN')
    # # plt.title(f'Distance Metrics for k-NN')

    # plt.title(f'Validation Curve for k-NN {param}')
    # plt.xlabel(f'{param}')
    # plt.ylabel(metric)
    # plt.legend(loc='best')
    # plt.grid()
    # plt.savefig(f'./plots/r_knn_{param}.png', dpi=300)

    # The test
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    if metric == 'accuracy':
        accuracy = accuracy_score(y_test, y_pred)
        print(f"KNN Test Accuracy: {accuracy:.2f}")
    if metric == 'recall_macro':
        recall = recall_score(y_test, y_pred, average='macro')
        print(f"KNN Test Recall: {recall:.2f}")