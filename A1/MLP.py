from sklearn.model_selection import learning_curve, validation_curve, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, log_loss
import numpy as np
import matplotlib.pyplot as plt


def NN(X_train, X_test, y_train, y_test, ax, metric):
    
    nn = MLPClassifier(
        random_state=90210, 
        hidden_layer_sizes=(200, 200, 200), 
        # max_iter=2000, 
        alpha=0.5,
        learning_rate_init=0.0001,
        early_stopping=True     # Need this for loss curve
    )

    ###################### Learning Curve ######################
    # nn.fit(X_train, y_train)

    # plt.clf()
    # plt.figure(figsize=(6,6)) # or (7,6)
    # plt.plot(nn.loss_curve_, label='Training Loss', color='blue')
    # plt.plot(nn.validation_scores_, label='Cross-Validation Loss', color='red', linestyle='--')
    # plt.title('Learning Curve for NN Classifier')
    # plt.xlabel('Epochs')
    # plt.ylabel("Loss")
    # plt.legend(loc='best')
    # plt.grid()
    # plt.savefig(f'./plots/r_nn_learn.png', dpi=300)

    ###################### Validation Curve ######################
    # param = 'hidden_layer_sizes'
    # param_range = [(100,), (100, 100), (100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100, 100)]  # (100, 100, 100)   # (100, 100, 100)
    # param = 'hidden_layer_sizes'
    # param_range = [(50, 50, 50), (100, 100, 100), (200, 200, 200), (300, 300, 300), (400, 400, 400)]      # (200, 200, 200)     # (300, 300, 300)
    # param = 'learning_rate'
    # param_range = ['constant', 'invscaling', 'adaptive']                          # any
    # param = 'alpha'
    # param_range = [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10]                       # 0.1
    # param = 'solver'
    # param_range = ['adam', 'sgd']                                                 # sgd 
    # param = 'learning_rate_init'
    # param_range = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]               # 0.0001    #1e-2
    # param = 'activation'
    # param_range = ['relu', 'identity', 'logistic', 'tanh']                          #           # identity

    # train_scores, valid_scores = validation_curve(
    #     nn,
    #     X_train, 
    #     y_train, 
    #     param_name=param,  
    #     param_range=param_range,
    #     cv=3,   
    #     scoring=metric,
    #     n_jobs=-1
    # )

    # train_mean = np.mean(train_scores, axis=1)
    # valid_mean = np.mean(valid_scores, axis=1)

    # plt.clf()
    # plt.figure(figsize=(6,6)) # or (7,6)

    # # # Depth
    # # plt.plot([len(x) for x in param_range], train_mean, label='Training Accuracy', color='blue', marker='o')
    # # plt.plot([len(x) for x in param_range], valid_mean, label='Cross-Validation Accuracy', color='green', marker='o')

    # # # Width
    # # plt.plot([x[0] for x in param_range], train_mean, label='Training Accuracy', color='blue', marker='o')
    # # plt.plot([x[0] for x in param_range], valid_mean, label='Cross-Validation Accuracy', color='green', marker='o')

    # plt.bar(param_range, train_mean, label='Training Accuracy', color='blue')
    # plt.bar(param_range, valid_mean, label='Validation Accuracy', color='green')
    # plt.title('NN Activation Functions')
    # plt.ylim([0.45, 0.9])


    # # plt.plot(param_range, train_mean, label='Training Accuracy', color='blue', marker='o')
    # # plt.plot(param_range, valid_mean, label='Cross-Validation Accuracy', color='green', marker='o')
    # # plt.title(f'Validation Curve for NN Classifier {param}')
    # plt.xlabel(f'{param} functions')
    # plt.ylabel(metric)
    # plt.legend(loc='best')
    # # plt.grid()
    # plt.savefig(f'./plots/r_nn_{param}.png', dpi=300)

    # The test
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)

    if metric == 'accuracy':
        accuracy = accuracy_score(y_test, y_pred)
        print(f"NN Test Accuracy: {accuracy:.2f}")
    if metric == 'recall_macro':
        recall = recall_score(y_test, y_pred, average='macro')
        print(f"NN Test Recall: {recall:.2f}")