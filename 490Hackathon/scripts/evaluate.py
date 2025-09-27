
#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bankruptcy.data import DEFAULT_FEATURES, NAME_COL, YEAR_COL, STATUS_COL

def pick_engine():
    try: import xlsxwriter; return "xlsxwriter"
    except Exception:
        try: import openpyxl; return "openpyxl"
        except Exception: return None

def read_any(path, sheet=None):
    p=Path(path); suf=p.suffix.lower()
    if suf in [".xlsx",".xls"]: return pd.read_excel(p, sheet_name=sheet)
    elif suf in [".csv",".txt"]: return pd.read_csv(p)
    else: raise ValueError("Use .xlsx/.xls/.csv")

def detect_binary_target(df):
    for c in ["Y","target","actual","label","y_true","y"]:
        if c in df.columns:
            vals=pd.to_numeric(df[c], errors="coerce").dropna().unique()
            if set(vals).issubset({0,1}): return c
    return None

def build_next_year_labels(df, target_key):
    for c in [NAME_COL, YEAR_COL, STATUS_COL]:
        if c not in df.columns: raise ValueError(f"Expected column '{c}'")
    d=df.copy(); d[STATUS_COL]=d[STATUS_COL].astype(str).str.strip().str.lower()
    d=d.sort_values([NAME_COL,YEAR_COL]); d["next_status"]=d.groupby(NAME_COL)[STATUS_COL].shift(-1)
    d=d[~d["next_status"].isna()].copy()
    if target_key=="Y_from_alive":
        d["Y_eval"]=((d[STATUS_COL]=="alive") & (d["next_status"]=="failed")).astype(int)
        d=d[d[STATUS_COL]=="alive"].copy()
    else:
        d["Y_eval"]=(d["next_status"]=="failed").astype(int)
    return d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", default="models/catboost_bankruptcy.cbm")
    ap.add_argument("--meta", default="models/metadata.json")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--out", default="scored_upload.xlsx")
    a=ap.parse_args()

    meta=json.loads(Path(a.meta).read_text())
    feats=meta["feature_cols"]; thr=float(a.threshold if a.threshold is not None else meta.get("threshold_f1",0.5))
    target_key=meta["target_col"]; task_desc=meta["task_desc"]

    clf=CatBoostClassifier(); clf.load_model(a.model)
    d=read_any(a.input, sheet=a.sheet); d.columns=d.columns.str.strip()
    tgt=detect_binary_target(d)
    df_eval=d.copy() if tgt else build_next_year_labels(d, target_key)
    if tgt: df_eval["Y_eval"]=pd.to_numeric(df_eval[tgt], errors="coerce").astype(int)

    missing=[c for c in feats if c not in df_eval.columns]
    if missing: raise ValueError(f"Missing features: {missing}")
    for c in feats: df_eval[c]=pd.to_numeric(df_eval[c], errors="coerce")
    mask=~np.isnan(df_eval[feats].to_numpy()).any(axis=1); df_eval=df_eval.loc[mask].copy()

    proba=clf.predict_proba(df_eval[feats].to_numpy())[:,1]; pred=(proba>=thr).astype(int); y_true=df_eval["Y_eval"].to_numpy()
    def safe(fn,*a): 
        try: return float(fn(*a))
        except: return None
    roc_auc=safe(roc_auc_score,y_true,proba); pr_auc=safe(average_precision_score,y_true,proba)
    cm=confusion_matrix(y_true,pred); tn,fp,fn,tp=(cm.ravel().tolist() if cm.size==4 else [None]*4)

    out=df_eval.copy()
    out["prob_fail_next_year"]=proba; out["pred_label"]=pred
    out["pred_status"]=np.where(pred==1,"Bankrupt (1)","Alive (0)"); out["threshold_used"]=thr

    out_path=Path(a.out)
    if out_path.suffix.lower() in [".xlsx",".xls"]:
        engine=pick_engine()
        if engine:
            with pd.ExcelWriter(out_path, engine=engine) as w:
                out.to_excel(w, index=False, sheet_name="scored")
                pd.DataFrame([{"task_desc":task_desc,"target_used":target_key,"threshold":thr,"n":len(out),
                               "positives":int(y_true.sum()),"roc_auc":roc_auc,"pr_auc":pr_auc,
                               "tn":tn,"fp":fp,"fn":fn,"tp":tp}]).to_excel(w,index=False,sheet_name="summary")
                try: pd.DataFrame(classification_report(y_true,pred,digits=4,output_dict=True)).T.to_excel(w,sheet_name="classification_report")
                except: pass
        else:
            out_path=out_path.with_suffix(".csv"); out.to_csv(out_path, index=False)
    else:
        out.to_csv(out_path, index=False)

    print("=== Batch Evaluation ===")
    print("Task:", task_desc, "| target:", target_key, "| threshold:", thr)
    print("Counts:", {"n": int(len(out)), "positives": int(y_true.sum())})
    print("ROC-AUC:", roc_auc, "PR-AUC:", pr_auc)
    if tn is not None:
        print("Confusion matrix [tn, fp, fn, tp]:", [tn, fp, fn, tp])
    print("Saved scored file:", str(out_path))

if __name__=="__main__": main()
