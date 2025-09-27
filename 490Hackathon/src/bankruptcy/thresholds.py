
from __future__ import annotations
import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix
def f1_opt_threshold(y_true, proba):
    prec,rec,thr=precision_recall_curve(y_true,proba); f1=2*(prec*rec)/(prec+rec+1e-12); import numpy as np
    return float(thr[np.nanargmax(f1[:-1])])
def choose_threshold_by_cost(y_true, proba, cost_fn=5.0, cost_fp=1.0):
    prec,rec,thr=precision_recall_curve(y_true,proba); best_t, best_cost=0.5, float("inf")
    for t in thr:
        yhat=(proba>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y_true,yhat).ravel(); cost=cost_fn*fn+cost_fp*fp
        if cost<best_cost: best_t, best_cost=float(t), float(cost)
    return best_t
