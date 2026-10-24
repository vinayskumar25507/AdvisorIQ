# checker.py
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

FEATURES = ["amount", "hour", "orig_error", "dest_error", "log_amount", "is_drain"]
MODEL_PATH = "models/checker.joblib"

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)
    df["hour"] = pd.to_numeric(df.get("hour", 12), errors="coerce").fillna(12)
    
    df["orig_error"] = (df["oldbalanceOrg"] - df["amount"]) - df["newbalanceOrig"]
    df["dest_error"] = (df["oldbalanceDest"] + df["amount"]) - df["newbalanceDest"]
    df["log_amount"] = np.log1p(df["amount"])
    
    # Deterministic feature
    df["is_drain"] = ((df["amount"] == df["oldbalanceOrg"]) & (df["amount"] > 0)).astype(int)
    return df

def train(csv="data/sample_transactions.csv"):
    df = pd.read_csv(csv)
    df_feat = engineer_features(df)
    X = df_feat[FEATURES].fillna(0)
    y = df["isFraud"].astype(int)
    
    # Split data to evaluate metrics robustly on unseen test records
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.sum() > 1 else None
    )
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42))
    ])
    model.fit(X_train, y_train)
    
    # Calculate evaluation metrics
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob) if len(set(y_test)) > 1 else 0.0
    
    print("========================================")
    print(" MODEL EVALUATION METRICS (checker.py)")
    print("========================================")
    print(f" Precision : {precision:.4f}")
    print(f" Recall    : {recall:.4f}")
    print(f" F1 Score  : {f1:.4f}")
    print(f" ROC-AUC   : {roc_auc:.4f}")
    print("========================================")
    
    # Retrain model on the full dataset for production/deployment usage
    model.fit(X, y)
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

def flag_transactions(df: pd.DataFrame):
    if not Path(MODEL_PATH).exists():
        train()
    model = joblib.load(MODEL_PATH)
    df_feat = engineer_features(df)
    X_pred = df_feat[FEATURES].fillna(0)
    
    scores = model.predict_proba(X_pred)[:, 1]
    df = df.assign(risk_score=scores)
    
    # HYBRID OVERRIDE: Evaluate against df_feat, apply to df
    df.loc[df_feat["is_drain"] == 1, "risk_score"] = 0.99
    
    return df[df.risk_score >= 0.85].sort_values("risk_score", ascending=False)

if __name__ == "__main__":
    train()