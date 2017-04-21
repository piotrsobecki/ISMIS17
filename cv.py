import pandas as pd
import numpy as np
from sklearn import metrics
from functools import partial

from commons import *
from classifier import *
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np


def crossvalidation_transformed(clf, X, Y, X_transformed, K=10):
    score = np.zeros(K)
    kf = StratifiedKFold(n_splits=K, shuffle=True)
    i = 0
    for train, test in kf.split(X, Y):
        print(i)
        X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], Y[train], Y[test]

        X_train_t = X_transformed[X_transformed[col_trans].isin(train)]
        X_test_t = X_transformed[X_transformed[col_trans].isin(test)]

        clf.fit_transformed(X_train_t)

        y_test_pred = clf.predict_transformed(X_test_t)
        print(set(y_test_pred))

        score[i] = accuracy(y_test, y_test_pred)
        i += 1
    return score


train = pd.DataFrame.from_csv(trainFile,sep=";",index_col=None)
test = pd.DataFrame.from_csv(testFile,sep=";",index_col=None)

train_t = pd.DataFrame.from_csv(trainTransformed,sep=";",index_col=None)
test_t = pd.DataFrame.from_csv(testTransformed,sep=";",index_col=None)


x_train = train
y_train = train[col_decision]





confidence_map = {
    CONF_NO_EXPERT_MODEL:0.5,
    CONF_NO_EXPERT_SYMBOL_MODEL:0.75,
    CONF_ALL:1
}
scores = crossvalidation_transformed( Classifier(confidence_map),x_train,y_train,train_t,3)
print('Final - Accuracy: %0.3f (+/- %0.3f)'%(np.mean(scores),scores.std()*2))


#confidence_map_all_same = {
#    CONF_NO_EXPERT_MODEL:1,
#    CONF_NO_EXPERT_SYMBOL_MODEL:1,
#    CONF_ALL:1
#}

#scores = crossvalidation_transformed( Classifier(confidence_map_all_same),x_train,y_train,train_t,3)
#print('All same - Accuracy: %0.3f (+/- %0.3f)'%(np.mean(scores),scores.std()*2))


#confidence_map_total_accuracy = {
#    CONF_NO_EXPERT_MODEL:0.5,
#    CONF_NO_EXPERT_SYMBOL_MODEL:1,
#    CONF_ALL:1
#}
#scores = crossvalidation_transformed( Classifier(confidence_map_total_accuracy),x_train,y_train,train_t,3)
#print('Accuracy: %0.3f (+/- %0.3f)'%(np.mean(scores),scores.std()*2))




#ex = Extractor()
#x_train_t = ex.transform(x_train)
#x_train_t[col_decision] = y_train[x_train_t[col_trans]].map(vmap).tolist()
#x_train_t.to_csv(trainTransformed,sep=";",index=False)
