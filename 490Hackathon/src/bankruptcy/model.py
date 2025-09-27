
from __future__ import annotations
import json
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from .thresholds import f1_opt_threshold, choose_threshold_by_cost
def train_catboost(X_train,y_train,X_val,y_val,random_state=42):
    import numpy as np
    n_pos=int((y_train==1).sum()); n_neg=int((y_train==0).sum()); spw=float(n_neg)/max(1.0,float(n_pos))
    clf=CatBoostClassifier(loss_function="Logloss",eval_metric="AUC",iterations=2000,learning_rate=0.05,depth=6,l2_leaf_reg=3.0,random_seed=random_state,scale_pos_weight=spw,verbose=200)
    clf.fit(X_train,y_train,eval_set=(X_val,y_val),use_best_model=True); return clf
def evaluate_thresholds(clf,X_val,y_val,cost_fn=5.0,cost_fp=1.0):
    import numpy as np
    p=clf.predict_proba(X_val)[:,1]; thr_f1=f1_opt_threshold(y_val,p); thr_cost=choose_threshold_by_cost(y_val,p,cost_fn,cost_fp)
    metrics={"val_roc_auc":float(roc_auc_score(y_val,p)),"val_pr_auc":float(average_precision_score(y_val,p)),"val_report_at_f1":classification_report(y_val,(p>=thr_f1).astype(int),digits=4)}
    return thr_f1,thr_cost,metrics
def save_artifacts(clf,out_model_path,meta_path,meta):
    clf.save_model(out_model_path); 
    with open(meta_path,"w") as f: json.dump(meta,f,indent=2)
