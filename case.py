import pandas as pd
import random

data = pd.read_csv("data/pos_neg_balance_data/result/data.csv")
data_unknown = pd.read_csv("data/pos_neg_balance_data/neg.csv")
data_unknown = data_unknown.drop("Unnamed: 0", axis=1)

data['random'] = 0
data['random'] = data['random'].apply(lambda x: random.randint(1, 20000))
data = data.sort_values(['random'])

X = data.drop("a", axis=1)
X = X.drop("b", axis=1)
X = X.drop("random", axis=1)
X = X.drop("target", axis=1)
y = data["target"].copy()

clf.fit(X, y)
pre_probas = clf.predict_proba(data_unknown)

probas = {}
for i in range(len(pre_probas)):
    probas[i] = pre_probas[i][1]
top = sorted(probas.items(), key=lambda x: x[1], reverse=True)[:200]
for each in top:
    print(each)
