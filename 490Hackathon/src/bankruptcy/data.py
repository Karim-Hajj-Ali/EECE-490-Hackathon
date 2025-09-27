
from __future__ import annotations
import pandas as pd, numpy as np
DEFAULT_FEATURES = [f"X{i}" for i in range(1, 19)]
NAME_COL, YEAR_COL, STATUS_COL = "company_name","year","status_label"
def load_csv(path): df=pd.read_csv(path); df.columns=df.columns.str.strip(); return df
def build_labels(df, feature_cols=DEFAULT_FEATURES):
    d=df.copy(); d[STATUS_COL]=d[STATUS_COL].astype(str).str.strip().str.lower()
    d=d.sort_values([NAME_COL,YEAR_COL]); d["next_status"]=d.groupby(NAME_COL)[STATUS_COL].shift(-1)
    d=d[~d["next_status"].isna()].copy()
    d["Y_any_next_fail"]=(d["next_status"]=="failed").astype(int)
    d["Y_from_alive"]=((d[STATUS_COL]=="alive") & (d["next_status"]=="failed")).astype(int)
    for c in feature_cols: d[c]=pd.to_numeric(d[c], errors="coerce")
    return d
def select_target(df):
    if int(df["Y_from_alive"].sum())>0: import pandas as pd; return "Y_from_alive","Predict onset: alive(t)->failed(t+1)", (df[STATUS_COL]=="alive")
    else: import pandas as pd; return "Y_any_next_fail","Predict failed(t+1) from any state", pd.Series(True, index=df.index)
def time_slice(d,y1,y2): return d[(d[YEAR_COL]>=y1)&(d[YEAR_COL]<=y2)].copy()
def split_by_time(df): return time_slice(df,1999,2011), time_slice(df,2012,2014), time_slice(df,2015,2018)
def arrays_from_frames(train_df,val_df,test_df,feature_cols,target_col):
    import numpy as np
    X_train,y_train=train_df[feature_cols].to_numpy(),train_df[target_col].to_numpy()
    X_val,y_val=val_df[feature_cols].to_numpy(),val_df[target_col].to_numpy()
    X_test,y_test=test_df[feature_cols].to_numpy(),test_df[target_col].to_numpy()
    def drop_nan(X,y): m=~np.isnan(X).any(axis=1); return X[m],y[m]
    return drop_nan(X_train,y_train)[0],drop_nan(X_train,y_train)[1],drop_nan(X_val,y_val)[0],drop_nan(X_val,y_val)[1],drop_nan(X_test,y_test)[0],drop_nan(X_test,y_test)[1]
