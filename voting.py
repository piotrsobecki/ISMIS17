import pandas as pd
import numpy as np
from sklearn import metrics
from functools import partial

from commons import *
from classifier import *
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np


def crossvalidation_transformed( X, Y, Y_pred, K=10):
    score = np.zeros(K)
    kf = StratifiedKFold(n_splits=K, shuffle=True)
    i = 0
    for train, test in kf.split(X, Y):
        print(i)
        X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], Y[train], Y[test]

        score[i] = accuracy(y_test, Y_pred[test])
        i += 1
    return score


train = pd.DataFrame.from_csv(trainFile,sep=";",index_col=None)
test = pd.DataFrame.from_csv(testFile,sep=";",index_col=None)

train_t = pd.DataFrame.from_csv(trainTransformed,sep=";",index_col=None)
test_t = pd.DataFrame.from_csv(testTransformed,sep=";",index_col=None)


x_train = train
y_train = train[col_decision]

groupped = train_t.groupby(["TransID",'Prediction'])['ExpertID'].agg('count')

dict = defaultdict(dict)

for name in groupped.index:
    trans_id, prediction = name
    dict[trans_id][prediction] =  groupped.loc[name]


y_pred = {}
for trans_id, predictions in dict.items():
    y_pred[trans_id] = vmap_rev[max(predictions, key=predictions.get)]

y_pred_labels = np.array(list(y_pred.values()))


scores = crossvalidation_transformed(x_train,y_train,y_pred_labels,3)


print('Final - Accuracy: %0.3f (+/- %0.3f)'%(np.mean(scores),scores.std()*2))