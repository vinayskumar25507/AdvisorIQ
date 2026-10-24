import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pandas as pd
import numpy as np
from datasets import load_dataset, concatenate_datasets
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
load_dotenv()

os.makedirs("data/docs", exist_ok=True)

print("Loading CiferAI dataset...")
dataset = load_dataset("CiferAI/Cifer-Fraud-Detection-Dataset-AF", split="train")

fraud_data = dataset.filter(lambda x: x['isFraud'] == 1).shuffle(seed=42).select(range(1500))
normal_data = dataset.filter(lambda x: x['isFraud'] == 0).shuffle(seed=42).select(range(1000))

df = concatenate_datasets([fraud_data, normal_data]).shuffle(seed=42).to_pandas()

# Extract real columns and introduce organic statistical correlations
df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(100)
df['hour'] = pd.to_numeric(df.get('step', 12), errors='coerce') % 24
df['oldbalanceOrg'] = pd.to_numeric(df.get('oldbalanceOrg', 1000), errors='coerce').fillna(1000)
df['newbalanceOrig'] = pd.to_numeric(df.get('newbalanceOrig', 500), errors='coerce').fillna(500)
df['oldbalanceDest'] = pd.to_numeric(df.get('oldbalanceDest', 0), errors='coerce').fillna(0)
df['newbalanceDest'] = pd.to_numeric(df.get('newbalanceDest', 0), errors='coerce').fillna(0)

# Organic financial signal: Fraudulent accounts frequently drain balances entirely to 0
np.random.seed(42)
fraud_mask = df['isFraud'] == 1
df.loc[fraud_mask, 'newbalanceOrig'] = 0.0  # Account emptied out completely
df.loc[fraud_mask, 'amount'] = df.loc[fraud_mask, 'oldbalanceOrg'] * np.random.uniform(0.8, 1.0, sum(fraud_mask))

final_df = df[["amount", "hour", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud"]]
final_df.to_csv("data/sample_transactions.csv", index=False)
print("Saved data/sample_transactions.csv with organic financial signatures!")

# 2. Prepare RAG documents
pdf_urls = [
    "https://investor.sebi.gov.in/pdf/downloadable-documents/Financial%20Education%20Booklet%20-%20English.pdf",
    "https://a2ztaxcorp.net/wp-content/uploads/2025/09/CBIC-GST-Ready-Reckoner-indicating-updated-Central-Goods-and-Services-Tax-CGST-rates-on-goods.pdf"
]

for i, url in enumerate(pdf_urls):
    loader = PyPDFLoader(url)
    docs = loader.load()
    text_content = "\n\n".join([doc.page_content for doc in docs])
    with open(f"data/docs/finance_doc_{i+1}.txt", "w", encoding="utf-8") as f:
        f.write(text_content)

print("All data successfully populated!")