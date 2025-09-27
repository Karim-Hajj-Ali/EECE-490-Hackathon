
from __future__ import annotations
import json, numpy as np, gradio as gr
from catboost import CatBoostClassifier
def load_model_and_meta(model_path, meta_path):
    m=CatBoostClassifier(); m.load_model(model_path)
    meta=json.loads(open(meta_path,"r").read())
    if "threshold_cost" not in meta: meta["threshold_cost"]=meta.get("threshold_f1", meta.get("threshold", 0.5))
    return m, meta
def build_gui(model_path, meta_path):
    m, meta = load_model_and_meta(model_path, meta_path)
    feat_cols=meta["feature_cols"]; thr_f1=float(meta.get("threshold_f1",0.5)); thr_cost=float(meta.get("threshold_cost",thr_f1))
    task_desc=meta.get("task_desc","Predict failed next year")
    med=meta.get("train_feature_medians",{}); def_vals=[float(med.get(c,0.0)) for c in feat_cols]
    def predict_one(*vals, threshold_mode, custom_threshold):
        x=np.array(vals,dtype=float).reshape(1,-1)
        thr = thr_f1 if threshold_mode=="Validation (F1-opt)" else (thr_cost if threshold_mode=="Cost-sensitive (FN>FP)" else float(custom_threshold))
        prob=float(m.predict_proba(x)[0,1]); pred=int(prob>=thr); status="Bankrupt (1)" if pred==1 else "Alive (0)"
        return {"Task":task_desc,"Probability: failed next year (t+1)":round(prob,4),"Decision threshold":round(thr,3),"Predicted label":int(pred),"Predicted status":status}
    inputs=[gr.Number(label=c, value=v) for c,v in zip(feat_cols,def_vals)]
    thr_mode=gr.Radio(choices=["Validation (F1-opt)","Cost-sensitive (FN>FP)","Custom"], value="Validation (F1-opt)", label="Decision rule")
    thr_slider=gr.Slider(0.0,1.0,value=thr_f1,step=0.001,label="Custom threshold")
    return gr.Interface(fn=predict_one, inputs=inputs+[thr_mode,thr_slider], outputs=gr.JSON(label="Prediction"),
                        title="Bankruptcy Risk (Next-Year) — Predictor", description=f"This model uses target: **{task_desc}**.", allow_flagging="never")
