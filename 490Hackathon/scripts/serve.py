#!/usr/bin/env python
from __future__ import annotations
import argparse, json, os
import numpy as np, gradio as gr
from catboost import CatBoostClassifier

def main():
    ap = argparse.ArgumentParser(description="Serve manual-input GUI (X1..X18 + threshold slider)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--meta",  required=True)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--inbrowser", action="store_true", help="Open browser automatically (works when not in Docker)")
    args = ap.parse_args()

    with open(args.meta, "r") as f:
        meta = json.load(f)

    feat_cols   = meta["feature_cols"]
    thr_default = float(meta.get("threshold_f1", 0.5))
    task_desc   = meta.get("task_desc", "Predict failed next year")
    med         = meta.get("train_feature_medians", {})

    m = CatBoostClassifier(); m.load_model(args.model)

    def predict_one(*vals, threshold=thr_default):
        x = np.array(vals, dtype=float).reshape(1, -1)
        prob = float(m.predict_proba(x)[0, 1])
        pred = int(prob >= threshold)
        return {
            "Task": task_desc,
            "Probability: failed next year (t+1)": round(prob, 4),
            "Threshold": round(float(threshold), 3),
            "Predicted label": int(pred),
            "Predicted status": "Bankrupt (1)" if pred == 1 else "Alive (0)"
        }

    inputs = [gr.Number(label=c, value=float(med.get(c, 0.0))) for c in feat_cols]
    thr_slider = gr.Slider(0.0, 1.0, value=thr_default, step=0.001, label="Decision threshold")

    demo = gr.Interface(
        fn=predict_one,
        inputs=inputs + [thr_slider],
        outputs=gr.JSON(label="Prediction"),
        title="Bankruptcy Risk (Next-Year) — Predictor",
        description=f"This model uses target: **{task_desc}**.\n"
                    "If your dataset lacks alive→failed transitions, we forecast failure next year from any state.",
        flagging_mode="never"  # use flagging_mode instead of allow_flagging
    )

    # Use env vars if present (Dockerfile sets GRADIO_SERVER_NAME=0.0.0.0 and port 7860)
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    # If you really want localhost when running natively, run with:  GRADIO_SERVER_NAME=localhost
    demo.launch(share=args.share, debug=False, server_name=server_name, server_port=server_port, inbrowser=args.inbrowser)

if __name__ == "__main__":
    main()
