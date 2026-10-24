# checker.py
import pandas as pd
import joblib
from pathlib import Path
import numpy as np

FEATURES = ["amount", "hour", "orig_error", "dest_error", "log_amount"]
MODEL_PATH = "models/checker.joblib"

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)
    
    if "step" in df.columns and "hour" not in df.columns:
        df["hour"] = pd.to_numeric(df["step"], errors="coerce").fillna(12) % 24
    else:
        df["hour"] = pd.to_numeric(df.get("hour", 12), errors="coerce").fillna(12)
        
    df["orig_error"] = (df["oldbalanceOrg"] - df["amount"]) - df["newbalanceOrig"]
    df["dest_error"] = (df["oldbalanceDest"] + df["amount"]) - df["newbalanceDest"]
    df["log_amount"] = np.log1p(df["amount"])
    
    return df

def flag_transactions(df: pd.DataFrame):
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError("Model file missing. Run local_train.py before deploying.")
        
    model = joblib.load(MODEL_PATH)
    df_feat = engineer_features(df)
    X_pred = df_feat[FEATURES].fillna(0)
    
    predictions = model.predict(X_pred)
    raw_scores = model.decision_function(X_pred)
    
    df = df.assign(risk_score=np.clip(0.5 - raw_scores, 0, 1))
    df = df.assign(is_anomaly=predictions)
    
    return df[df.is_anomaly == -1].sort_values("risk_score", ascending=False)