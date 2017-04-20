from commons import *
from extractors import *
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np


class ExpertModel:
    def __init__(self, model):
        self.model = model

    def recall(self, case):
        return self.model[rmap[case]]

    def precision(self, case):
        return self.model[pmap[case]]

    def n(self, case):
        return self.model[nmap[case]]

class DefaultExpertModel:
    def recall(self, case):
        return 0.5

    def precision(self, case):
        return 0.5

    def n(self, case):
        return 1

class ExpertClassifier:
    def __init__(self, model):
        self.model = model

    def get_proba(self, pred):
        proba = {}
        proba[pred] = self.model.precision(pred)
        s_recall = 0
        for option in (opt for opt in [buy, hold, sell] if opt is not pred):
            s_recall = s_recall + (1 - self.model.recall(option))

        for option in (opt for opt in [buy, hold, sell] if opt is not pred):
            proba[option] = ((1 - proba[pred]) * (1 - self.model.recall(option))) / s_recall

        return proba

    def get_model(self):
        return self.model

class DefaultExpertClassifier:
    def get_proba(self, pred):
        proba = {}
        proba[pred] = 1
        for option in (opt for opt in [buy, hold, sell] if opt is not pred):
            proba[option] = 0
        return proba

    def get_model(self):
        return DefaultExpertModel()



class Classifier:
    def __init__(self, base_proba, confidence_map):
        self.base_proba = base_proba
        self.confidence_map = confidence_map

    def fit(self, X_train, y_train):
        ex = Extractor()
        t_train = ex.transform(X_train)
        t_train[col_decision] = y_train[t_train[col_trans]].map(vmap).tolist()

        self.experts = ex.extract_experts(t_train)
        self.experts_symbols = ex.extract_expert_symbols(t_train)

    def fit_transformed(self, t_train):
        ex = Extractor()
        self.experts = ex.extract_experts(t_train)
        self.experts_symbols = ex.extract_expert_symbols(t_train)

    def predict(self, X_test):
        ex = Extractor()
        t_test = ex.transform(X_test)
        probas = self.extract_probas(t_test)
        features = ex.extract_features(probas,self.confidence_map)
        return classify(features)

    def predict_transformed(self, t_test):
        ex = Extractor()
        probas = self.extract_probas(t_test)
        features = ex.extract_features(probas,self.confidence_map)
        return classify(features)

    def extract_probas(self, data):
        probas = defaultdict(list)
        lab = []
        count = 0
        expertMiss = 0
        expertSymbolMiss = 0
        for ns, bytrans in data.groupby(col_trans):
            for eid, byexpert in bytrans.groupby(col_expert):
                byexpert = byexpert.sort(['TimeBefore'])
                symbol = byexpert[col_symbol].iloc[0]
                trans = byexpert[col_trans].iloc[0]
                pred = int(byexpert[col_prediction].iloc[0])
                # Expert Model
                clf = DefaultExpertClassifier()
                confidence = CONF_NO_EXPERT_MODEL
                count = count + 1
                expert_symbol_df = self.experts_symbols[
                    (self.experts_symbols[col_expert] == eid) & (self.experts_symbols[col_symbol] == symbol)]
                if len(expert_symbol_df) > 0:
                    expert = expert_symbol_df.iloc[0]
                    expert_model = ExpertModel(expert.to_dict())
                    clf = ExpertClassifier(expert_model)
                    confidence = CONF_ALL
                else:
                    expertSymbolMiss = expertSymbolMiss + 1
                    # Expert Model
                    expert_df = self.experts[self.experts[col_expert] == eid]
                    if len(expert_df) > 0:
                        expert = expert_df.iloc[0]
                        expert_model = ExpertModel(expert.to_dict())
                        clf = ExpertClassifier(expert_model)
                        confidence = CONF_NO_EXPERT_SYMBOL_MODEL
                    else:
                        expertMiss = expertMiss + 1

                for key, val in clf.get_proba(pred).items():
                    probas[str(key)].append(val)
                    probas[str(key)+'_n'].append(clf.get_model().n(key))

                probas[col_conf].append(confidence)
                probas[col_trans].append(ns)
                probas[col_symbol].append(symbol)
                probas[col_expert].append(eid)
        # print('Expert Miss {} / {}'.format(expertMiss,count))
        # print('Expert Symbol Miss {} / {}'.format(expertSymbolMiss,count))
        features = pd.DataFrame(probas)
        features = features.fillna(0)
        return features



def acc(matrix):
    cost = np.array([[8, 4, 8], [1, 1, 1], [8, 4, 8]])
    x = np.sum(matrix.diagonal() * cost.diagonal())
    y = np.sum(matrix * cost)
    return float(x) / float(y)


def accuracy(y_true, y_pred):
    cm = confusion_matrix(y_pred, y_true)
    print(cm)
    return acc(cm)


def normalize(trans_feat, pre):
    pre_sum = 0
    for key, val in trans_feat.items():
        if key.startswith(pre):
            pre_sum += val

    for key, val in trans_feat.items():
        if key.startswith(pre):
            trans_feat[key + "_norm"] = val / pre_sum


def resultant(x, y, z):
    return (x * -1 + y * 0 + z * 1) / (x + y + z)


def classify(features):
    classified = features[[vmap_rev[buy], vmap_rev[hold], vmap_rev[sell]]].idxmax(axis=1)
    return [(clas if clas and str(clas)!='nan' else vmap_rev[hold]) for clas in classified]

def extract_labels(data):
    lab = []
    for ns, bytrans in data.groupby(col_trans):
        lab.append(vmap_rev[bytrans['Decision'].iloc[0]])
    return lab
