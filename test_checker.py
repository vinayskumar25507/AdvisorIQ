import pandas as pd
from checker import engineer_features, flag_transactions

def test_engineer_features_is_drain():
    # Simulate a full account drain
    data = pd.DataFrame({
        "amount": [950.0],
        "oldbalanceOrg": [950.0],
        "newbalanceOrig": [0.0],
        "oldbalanceDest": [100.0],
        "newbalanceDest": [1050.0],
        "hour": [3]
    })
    
    df_feat = engineer_features(data)
    
    # Verify the engineering logic correctly flags the drain
    assert df_feat.iloc[0]["is_drain"] == 1

def test_flag_transactions_hybrid_override():
    # Simulate a critical anomaly that must trigger the override
    data = pd.DataFrame({
        "amount": [95000.0],
        "oldbalanceOrg": [95000.0],
        "newbalanceOrig": [0.0],
        "oldbalanceDest": [0.0],
        "newbalanceDest": [95000.0],
        "hour": [14]
    })
    
    result_df = flag_transactions(data)
    
    # Verify the final risk score is strictly overridden to 0.99
    assert not result_df.empty
    assert result_df.iloc[0]["risk_score"] == 0.99