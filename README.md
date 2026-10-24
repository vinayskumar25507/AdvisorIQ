# AdvisorIQ | AI for Finance 

AdvisorIQ is an enterprise-grade wealth management assistant and risk intelligence terminal. This capstone project integrates a modern web interface, large language models, and traditional machine learning to provide grounded financial insights and detect fraudulent transaction patterns.

### Key Features
* **Custom Fintech Terminal:** A lightweight, minimalist web UI built with FastAPI, vanilla JavaScript, and Tailwind CSS, featuring dark/light mode and dynamic multi-file CSV uploads.
* **Generative AI Agent:** Powered by Gemini 3.5 via LangChain's native `bind_tools`, the agent intelligently routes user queries to live market data, semantic document search, or anomaly detection tools.
* **Organic Risk Scoring:** A robust `scikit-learn` Random Forest pipeline (`checker.py`) that analyzes financial ledgers for account-drainage behaviors and anomalies, achieving highly realistic ROC-AUC scores.
* **Retrieval-Augmented Generation (RAG):** Local semantic search using ChromaDB and HuggingFace sentence transformers to ground AI responses in actual compliance and financial education documents.
* **Enterprise Guardrails:** Strict PII redaction (account numbers, emails, phone numbers) and compliance filters that block over-promising financial phrases.

### Tech Stack
* **Backend:** FastAPI, Uvicorn, Python
* **AI & Machine Learning:** LangChain, Google Gemini 3.5, scikit-learn, HuggingFace, ChromaDB
* **Data Processing:** pandas, numpy, yfinance
* **Frontend:** HTML5, Tailwind CSS, Vanilla JS

### Local Setup & Execution
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file in the root directory and add your Google API key: `GOOGLE_API_KEY="your_key"`
4. Build the RAG vector database: `python rag.py`
5. Generate the machine learning risk model and transaction data: `python setup_data.py`
6. Launch the application: `python app.py`
7. Access the terminal at `http://127.0.0.1:7860`.

### Deployment
This application is configured for seamless deployment on Render using standard Uvicorn commands.