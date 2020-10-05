from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

data = load_breast_cancer()
X = data["data"]
y = data["target"]

#pd.DataFrame(data=X).to_csv("data.csv")
#pd.DataFrame(data=y).to_csv("target.csv")

data_clean = pd.read_csv("data_clean.csv")
data_noisy = pd.read_csv("data_noisy.csv")

X_clean = data_clean.drop("target", axis=1)
y_clean = data_clean["target"]
X_noisy = data_noisy.drop("target", axis=1)
y_noisy = data_noisy["target"]

clf = LogisticRegression(random_state=123)

for scoring in ['roc_auc', 'average_precision', 'precision', 'recall', 'f1']:
    print(scoring)
    print(cross_val_score(clf, X_clean, y_clean, cv=5, scoring=scoring).mean())
    print(cross_val_score(clf, X_noisy, y_noisy, cv=5, scoring=scoring).mean())