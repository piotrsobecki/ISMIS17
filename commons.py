col_expert = "ExpertID"
col_symbol = "SymbolID"
col_prediction = "Prediction"
col_decision = "Decision"
col_trans = "TransID"
col_conf = 'Confidence'
col_n = 'N'

trainFile = "./train/trainingData.csv"
testFile = "./test/testData.csv"
trainTransformed = "./train/transformed.csv"
testTransformed = "./test/transformed.csv"

trainTransformedProbas = "./train/transformed-probas.csv"
testTransformedProbas = "./test/transformed-probas.csv"

expertsFile = "./experts.csv"
expertsSymbolsFile = "./experts-symbols.csv"

out_features_test = "./features_test.csv"
out_features_train = "./features_train.csv"

CONF_NO_EXPERT_MODEL = 0
CONF_NO_EXPERT_SYMBOL_MODEL = 1
CONF_ALL=2

buy  = 1
hold = 0
sell = -1

vmap = {"Buy":buy,"Hold":hold,"Sell":sell}
vmap_rev = {y:x for x,y in vmap.items()}

t = 1
f = 0

buy_vmap =  {sell:f,hold:f,buy:t}
hold_vmap = {sell:f,hold:t,buy:f}
sell_vmap = {sell:t,hold:f,buy:f}

nmap = {
    buy:"Buy_n",
    hold:"Hold_n",
    sell:"Sell_n"
}

nmap_norm = {
    buy:"Buy_n_norm",
    hold:"Hold_n_norm",
    sell:"Sell_n_norm"
}

pmap = {
    buy:"Buy_p",
    hold:"Hold_p",
    sell:"Sell_p"
}

rmap = {
    buy:"Buy_r",
    hold:"Hold_r",
    sell:"Sell_r"
}

cost_matrix = [
	[8,4,8],
	[1,1,1],
	[8,4,8]
]



