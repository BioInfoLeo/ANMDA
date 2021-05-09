#A part of code is deleted and it can be asked to reference by corresondence author after the paper is published.

import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
import random
import warnings
warnings.simplefilter('ignore')

data = pd.read_csv("data/pos_neg_balance_data/result/data.csv")

X = data.drop("a", axis=1)
X = X.drop("b", axis=1)
X = X.drop("target", axis=1)
y = data["target"].copy()


def plot_roc_curve(fpr, tpr, color, linestyle, label=None):
    plt.plot(fpr, tpr, c=color, ls=linestyle, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


def plot_presion_recall_curve(precisions, recalls, color, linestyle, label=None):
    plt.plot(recalls, precisions, c=color, ls=linestyle, linewidth=2, label=label)
    plt.plot([0, 1], [1, 0], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel("Recall")
    plt.ylabel("Precision")


def algorithms(curve):
    i = 0
    for clf in [clf]:
        y_probas = cross_val_predict(clf, X, y, cv=5, method="predict_proba")

        y_scores_tmp = []
        for each in y_probas:
            y_scores_tmp.append(each[1])
        y_scores = np.array(y_scores_tmp)

        if curve == 'roc':
            fpr, tpr, thresholds = roc_curve(y, y_scores)
            plot_roc_curve(fpr, tpr, show_algorithms_dict[i][0], show_algorithms_dict[i][1], show_algorithms_dict[i][2])
        if curve == 'pr':
            precisions, recalls, thresholds = precision_recall_curve(y, y_scores)
            plot_presion_recall_curve(precisions, recalls, show_algorithms_dict[i][0], show_algorithms_dict[i][1], show_algorithms_dict[i][2])

        print(i)
        print(roc_auc_score(y, y_scores))
        print(average_precision_score(y, y_scores))

        y_pre = []
        for each in y_scores:
            if each >= 0.5:
                y_pre.append(1)
            else:
                y_pre.append(0)

        print(precision_score(y, y_pre))
        print(recall_score(y, y_pre))
        print(f1_score(y, y_pre))

        i = i + 1
    plt.legend()
    plt.show()

random_data = data

score = []
for i in range(100):
    print(i)
    random_data['random'] = 0
    random_data['random'] = random_data['random'].apply(lambda x: random.randint(1, 20000))
    random_data = random_data.sort_values(['random'])

    X_random = random_data.drop(['a', 'b', 'target', 'random'], axis=1)
    y_random = random_data['target'].copy()

    for scoring in ['roc_auc']:
        score.append(cross_val_score(clf, X_random, y_random, cv=5, scoring=scoring).mean())
    pd.DataFrame(data=score).to_csv('score.csv')
