import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

FEATURES = ["amount", "hour", "orig_error", "dest_error", "log_amount"]
MODEL_PATH = "models/checker.joblib"

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["amount"] = pd.to_numeric(df.get("amount", 0), errors="coerce").fillna(0)
    df["hour"] = pd.to_numeric(df.get("hour", 12), errors="coerce").fillna(12)
    df["oldbalanceOrg"] = pd.to_numeric(df.get("oldbalanceOrg", 0), errors="coerce").fillna(0)
    df["newbalanceOrig"] = pd.to_numeric(df.get("newbalanceOrig", 0), errors="coerce").fillna(0)
    df["oldbalanceDest"] = pd.to_numeric(df.get("oldbalanceDest", 0), errors="coerce").fillna(0)
    df["newbalanceDest"] = pd.to_numeric(df.get("newbalanceDest", 0), errors="coerce").fillna(0)
    
    # Real financial anomaly indicators (discrepancies in account balances)
    df["orig_error"] = (df["oldbalanceOrg"] - df["amount"]) - df["newbalanceOrig"]
    df["dest_error"] = (df["oldbalanceDest"] + df["amount"]) - df["newbalanceDest"]
    df["log_amount"] = np.log1p(df["amount"])
    return df

def train(csv="data/sample_transactions.csv"):
    df = pd.read_csv(csv)
    df_feat = engineer_features(df)
    
    X = df_feat[FEATURES].fillna(0)
    y = df["isFraud"].astype(int)
    
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=250, 
            max_depth=12, 
            class_weight="balanced", 
            random_state=42
        ))
    ])
    
    model.fit(Xtr, ytr)
    
    preds = model.predict(Xte)
    probs = model.predict_proba(Xte)[:, 1]
    
    print("--- Organic Real-World Model Performance ---")
    print(classification_report(yte, preds))
    print("Organic ROC-AUC Score:", roc_auc_score(yte, probs))
    
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Production model saved to {MODEL_PATH}")

def flag_transactions(df: pd.DataFrame):
    if not Path(MODEL_PATH).exists():
        train()
    model = joblib.load(MODEL_PATH)
    df_feat = engineer_features(df)
    X_pred = df_feat[FEATURES].fillna(0)
    scores = model.predict_proba(X_pred)[:, 1]
    df = df.assign(risk_score=scores)
    return df[df.risk_score > 0.5].sort_values("risk_score", ascending=False)

if __name__ == "__main__":
    train()