import pandas as pd
import numpy as np
from sklearn import metrics 
from functools import partial
from collections import defaultdict
from scipy import stats
import math
from commons import *
from classifier import *

CONF_NO_EXPERT_MODEL = 0
CONF_NO_EXPERT_SYMBOL_MODEL = 1
CONF_ALL=2

class Extractor:
	def dfunc(self,func,grouped,filt):
		vals = []
		for name, group in grouped:
			vals.append(func(filt(group))) 
		return vals

	def mfunc(self,func,valmap,grouped):
		vals = []
		for name, group in grouped:
			dec = group[col_decision].map(valmap)
			pred = group[col_prediction].map(valmap)      
			vals.append(func(dec,pred)) 
		return vals

	def fdec(self,col,val,group):
		return group[group[col]==val]

		
	def transform(self,data):
		out = pd.DataFrame(columns=[col_trans,col_symbol,col_expert,col_prediction,"Rate","TimeBefore"])
		for idx,row in data.iterrows():
			sid = row[col_symbol]
			recommendations = row["Recommendations"]
			rec =  [val.replace("}","").replace("{","") for val in recommendations.split("}{")]
			rec = [[idx,sid] + val.split(',') for val in rec]
			for r in rec:
				out.loc[len(out)] = r
		out[col_prediction] = out[col_prediction].map(vmap)
		return out

	def extract_features(self, probas, confidence_map):
		x_probas = probas.copy()
		#print(x_probas)
		#raise "hepro"
		x_probas[col_conf] = x_probas[col_conf].map(confidence_map)
		n = x_probas[str(buy)+'_n'] +  x_probas[str(hold)+'_n'] +  x_probas[str(sell)+'_n']
		n.replace(0, 1, inplace=True)
		x_probas[buy] =  x_probas[str(buy)]  * x_probas[col_conf] * n**1/4
		x_probas[hold] = x_probas[str(hold)] * x_probas[col_conf] * n**1/4
		x_probas[sell] = x_probas[str(sell)] * x_probas[col_conf] * n**1/4
		# x_probas[buy]  = x_probas[str(buy)] * x_probas[col_conf]  + (1 - x_probas[col_conf]) * self.base_proba[buy]
		# x_probas[hold] = x_probas[str(hold)] * x_probas[col_conf] + (1 - x_probas[col_conf]) * self.base_proba[hold]
		# x_probas[sell] = x_probas[str(sell)] * x_probas[col_conf] + (1 - x_probas[col_conf]) * self.base_proba[sell]
		train_trans_group = x_probas.groupby(col_trans)
		return pd.DataFrame({
			vmap_rev[buy]: train_trans_group[buy].sum() /   ( train_trans_group[sell].sum() + train_trans_group[hold].sum() + train_trans_group[buy].sum()),
			vmap_rev[hold]: train_trans_group[hold].sum() / ( train_trans_group[sell].sum() + train_trans_group[hold].sum() + train_trans_group[buy].sum()),
			vmap_rev[sell]: train_trans_group[sell].sum() / ( train_trans_group[sell].sum() + train_trans_group[hold].sum() + train_trans_group[buy].sum()),
			col_conf: train_trans_group[col_conf].mean()
		}, columns=[vmap_rev[buy], vmap_rev[hold], vmap_rev[sell], col_conf])

	def extract_experts(self,t_train):
		x_group = t_train.groupby([col_expert])
		experts = pd.DataFrame(
			np.array([
				x_group[col_expert].first(),
				self.mfunc(metrics.precision_score,buy_vmap,x_group),
				self.mfunc(metrics.recall_score,buy_vmap,x_group),
				self.dfunc(len,x_group,partial(self.fdec,col_decision,buy)),
				self.mfunc(metrics.precision_score,hold_vmap,x_group),
				self.mfunc(metrics.recall_score,hold_vmap,x_group),
				self.dfunc(len,x_group,partial(self.fdec,col_decision,hold)),
				self.mfunc(metrics.precision_score,sell_vmap,x_group),
				self.mfunc(metrics.recall_score,sell_vmap,x_group),
				self.dfunc(len,x_group,partial(self.fdec,col_decision,sell))
			]).T,
			columns=[
				col_expert,
				pmap[buy],
				rmap[buy],
				nmap[buy],
				pmap[hold],
				rmap[hold],
				nmap[hold],
				pmap[sell],
				rmap[sell],
				nmap[sell]
			]
		) 
		experts[nmap_norm[buy]]  = 100 * experts[nmap[buy]] / sum(experts[nmap[buy]])
		experts[nmap_norm[hold]] = 100 * experts[nmap[hold]] / sum(experts[nmap[hold]])
		experts[nmap_norm[sell]] = 100 * experts[nmap[sell]] / sum(experts[nmap[sell]])
		return experts
		
	def extract_expert_symbols(self,t_train):
		x_group = t_train.groupby([col_expert,col_symbol])
		experts = pd.DataFrame(
			np.array([
				x_group[col_expert].first(),
				x_group[col_symbol].first(),
				self.mfunc(metrics.precision_score,buy_vmap,x_group),
				self.mfunc(metrics.recall_score,buy_vmap,x_group),
				self.dfunc(len,x_group,partial(self.fdec,col_decision,buy)),
				self.mfunc(metrics.precision_score,hold_vmap,x_group),
				self.mfunc(metrics.recall_score,hold_vmap,x_group),
				self.dfunc(len,x_group,partial(self.fdec,col_decision,hold)),
				self.mfunc(metrics.precision_score,sell_vmap,x_group),
				self.mfunc(metrics.recall_score,sell_vmap,x_group),
				self.dfunc(len,x_group,partial(self.fdec,col_decision,sell))
			]).T,
			columns=[
				col_expert,
				col_symbol,
				pmap[buy],
				rmap[buy],
				nmap[buy],
				pmap[hold],
				rmap[hold],
				nmap[hold],
				pmap[sell],
				rmap[sell],
				nmap[sell]
			]
		) 
		experts[nmap_norm[buy]]  = 100 * experts[nmap[buy]] / sum(experts[nmap[buy]])
		experts[nmap_norm[hold]] = 100 * experts[nmap[hold]] / sum(experts[nmap[hold]])
		experts[nmap_norm[sell]] = 100 * experts[nmap[sell]] / sum(experts[nmap[sell]])
		return experts
		