import os
import time
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load the local .env file so Google API can authenticate
load_dotenv()

CHROMA_DIR = "chroma_store"

# Lightweight Google API Embeddings
EMBEDDINGS = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def build_library(docs_dir="data/docs"):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs = [
        Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": str(p)})
        for p in sorted(Path(docs_dir).glob("**/*")) if p.is_file()
    ]
    if not docs:
        raise SystemExit(f"No documents found in {docs_dir!r} - add some .txt/.md files and re-run.")
        
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
    print(f"Total chunks to process: {len(chunks)}")
    
    # Initialize the vector store
    store = Chroma(persist_directory=CHROMA_DIR, embedding_function=EMBEDDINGS)
    
    # Batch chunks to respect Google's Free Tier limit (100 RPM)
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        store.add_documents(batch)
        print(f"Indexed chunks {i} to {i + len(batch)}")
        
        # Wait 60 seconds before the next batch to reset the quota limit
        if i + batch_size < len(chunks):
            print("Waiting 60 seconds to respect API rate limits...")
            time.sleep(60)
            
    print("Database built successfully!")

def get_retriever(k=4):
    store = Chroma(persist_directory=CHROMA_DIR, embedding_function=EMBEDDINGS)
    return store.as_retriever(search_kwargs={"k": k})

if __name__ == "__main__":
    build_library()