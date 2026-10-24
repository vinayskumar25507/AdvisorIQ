# local_train.py
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

FEATURES = ["amount", "hour", "orig_error", "dest_error", "log_amount"]
MODEL_PATH = "models/checker.joblib"

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)
    
    # PaySim uses 'step' for hours; map it if present
    if "step" in df.columns and "hour" not in df.columns:
        df["hour"] = pd.to_numeric(df["step"], errors="coerce").fillna(12) % 24
    else:
        df["hour"] = pd.to_numeric(df.get("hour", 12), errors="coerce").fillna(12)
        
    df["orig_error"] = (df["oldbalanceOrg"] - df["amount"]) - df["newbalanceOrig"]
    df["dest_error"] = (df["oldbalanceDest"] + df["amount"]) - df["newbalanceDest"]
    df["log_amount"] = np.log1p(df["amount"])
    
    return df

if __name__ == "__main__":
    print("Loading 6M row PaySim dataset (this may take a moment)...")
    df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")
    
    df_feat = engineer_features(df)
    X = df_feat[FEATURES].fillna(0)
    
    print("Training IsolationForest... ")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", IsolationForest(
            n_estimators=100, 
            contamination=0.001,  # 0.1% true outlier rate for production
            n_jobs=-1,            # Use all local CPU cores
            random_state=42
        ))
    ])
    
    model.fit(X)
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Production model successfully trained and saved to {MODEL_PATH}")