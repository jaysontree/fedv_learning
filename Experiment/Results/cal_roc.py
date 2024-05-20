
"""
> Yue Yijie, jaysonyue@outlook.sg
> 20240515
used for threshold search and ROC graph
This Confusion Matrix is calculated for DIAGNOSE RESULTS! please use original YOLO metrics to evaluate model.
"""
import os
import sys
import json
from collections import OrderedDict
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc
from matplotlib import pyplot

def _load_gt(path: str) -> dict:
	gts = dict()
	for file in os.listdir(path):
		lines = open(os.path.join(path, file),'r').readlines()
		if lines:
			gt_lab = lines[0].strip().split()[0]
			gts[file.split('.')[0]] = int(gt_lab)
	return gts

def _load_pred(path: str) -> dict:
	preds = dict()
	"""
	# val json results
	res = json.load(open(path,'r'))
	preds[i['image_id']]=(i['category_id'], i['score'])
	"""
	# pred text results
	for file in os.listdir(path):
		pred_info = open(os.path.join(path, file),'r').readlines()
		if pred_info:
			idx, score = int(pred_info[0].split()[0]), float(pred_info[0].split()[-1])
			preds[file.split('.')[0]] = (idx, score)

	return preds


def calculate_roc(_preds: dict, _gts: dict) -> list:

	# If true value is POSITIVE, and pred is negative or background, consider as FN
	# If true value is NEGATIVE, and pred is positive, consider as FP
	# If background, and pred is positive, consider as FP, no such case in test set.
	# TP: true value is POSITIVE and pred is positive
	# TN: true value is NEGATIVE and pred is not positive

	keys = list(_preds.keys())
	scores = [x[1] for k,x in _preds.items()]
	sort_idx = np.argsort(scores)[::-1]

	# search conf threshold from 1 to 0
	# at the beginning conf:1 all is predicted as negative
	N = len([k for k,v in _gts.items() if v != 1])
	P = len([k for k,v in _gts.items() if v == 1]) 
	TN = N
	FN = P
	TP = 0
	FP = 0

	_fpr = [0]
	_tpr = [0]

	results = []
	for idx in sort_idx:
		conf = scores[idx]
		temp_key = keys[idx]
		pred = _preds[temp_key][0]
		gt = _gts[temp_key]
		print (conf, temp_key, pred, gt)
		if gt == pred == 1: # true label is Positive and pred is positive
			TP += 1
			FN -= 1
		elif gt == 0 and pred == 1: # true label is Negative and positive
			FP += 1
			TN -= 1
		# in case pred == 0, just pass. if label==0, it is TN. if label==1 it is FN.
		sensitivity = TP / P
		specificity = TN / N
		TPR = TP / P 
		FPR = FP / N
		YOUDEN_INDEX = TPR - FPR
		ACC = (TP + TN) / (P + N)
		PRECISION = TP / (TP + FP + 1e-7)
		RECALL = TP / P
		_fpr.append(FPR)
		_tpr.append(TPR)
		results.append([conf, TP, FN, FP, TN, ACC, FPR, TPR, sensitivity, specificity, YOUDEN_INDEX, PRECISION, RECALL, pred, gt])


	_auc = auc(_fpr, _tpr)
	pyplot.figure()
	pyplot.plot(_fpr, _tpr, color="orange", lw=2, label=f'ROC curve (AUC={round(_auc, 2)})')
	pyplot.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
	pyplot.xlim([0.0, 1.0])
	pyplot.ylim([0.0, 1.05])
	pyplot.xlabel("FPR")
	pyplot.ylabel("TPR")
	pyplot.legend(loc="lower right")
	pyplot.savefig('roc.jpg')
	return results


if __name__=="__main__":
	gt_path = sys.argv[1]
	pred_path = sys.argv[2]
	res = calculate_roc(_load_pred(pred_path), _load_gt(gt_path))
	with open('res.csv','w') as f:
		f.write(f"conf,tp,fn,fp,tn,acc,fpr,tpr,sensitivity,specificity,youden,precision, recall, pred, gt\n")
		for r in res:
			f.write(",".join([str(i) for i in r]) + '\n')
