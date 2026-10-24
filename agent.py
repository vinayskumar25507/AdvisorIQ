# agent.py
import os
import shutil
from pathlib import Path
import pandas as pd
from langchain_core.tools import Tool
from langchain_core.messages import ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from rag import get_retriever
from checker import flag_transactions
from guardrails import make_safe

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key,
    temperature=0.0,
    # Or implement this instruction via the standard LangChain PromptTemplate
    system_instruction="You are a strict financial compliance AI. Summarize the provided ledger anomalies. Do not provide financial advice. If no anomalies are provided in the context, explicitly state 'I have reviewed your ledger, and no suspicious transactions were found.' and output nothing else."
)
retriever = get_retriever()
UPLOAD_PATH = "data/_current_upload.csv"

def market_data(ticker: str) -> str:
    import yfinance as yf
    try:
        hist = yf.Ticker(ticker.strip().upper()).history(period="1mo")
        if hist.empty:
            return f"No market data available for {ticker}."
        price, month_ago = hist["Close"].iloc[-1], hist["Close"].iloc[0]
        change = (price - month_ago) / month_ago * 100
        return f"{ticker}: price ${price:.2f}, 1-month change {change:+.1f}%."
    except Exception as e:
        return f"Market data service temporarily unavailable for {ticker}."

def research(query: str) -> str:
    hits = retriever.invoke(query)
    return "\n\n".join(f"[source] {d.page_content}" for d in hits) or "No relevant documents."

def check_uploaded(_input: str) -> str:
    if not Path(UPLOAD_PATH).exists():
        return "No transactions were uploaded for this question."
    flagged = flag_transactions(pd.read_csv(UPLOAD_PATH))
    if flagged.empty:
        return "No suspicious transactions found."
    return "Suspicious transactions:\n" + flagged.to_string(index=False)

TOOLS = [
    Tool(name="MarketData", func=market_data, description="Get live price and 1-month trend for a stock ticker like AAPL."),
    Tool(name="Research", func=research, description="Look up finance documents to ground an answer about a company or topic."),
    Tool(name="CheckTransactions", func=check_uploaded, description="Scan the user's uploaded transactions for suspicious activity."),
]

llm_with_tools = llm.bind_tools(TOOLS)

def run_advisor(question: str, csv_path: str | None = None) -> str:
    if csv_path:
        shutil.copyfile(csv_path, UPLOAD_PATH)
    else:
        Path(UPLOAD_PATH).unlink(missing_ok=True)

    messages = [
        ("system", "You are AdvisorIQ, a wealth-management assistant. Use the tools when needed to ground your answers in real data. Be concise and never promise a specific return."),
        ("human", question)
    ]
    
    response = llm_with_tools.invoke(messages)
    
    if response.tool_calls:
        tool_map = {t.name: t for t in TOOLS}
        tool_messages = [response]
        for call in response.tool_calls:
            tool_fn = tool_map.get(call["name"])
            if tool_fn:
                tool_args = call.get("args", {})
                arg_val = list(tool_args.values())[0] if tool_args else ""
                tool_output = tool_fn.func(str(arg_val))
                tool_messages.append(ToolMessage(content=tool_output, tool_call_id=call["id"]))
        
        final_response = llm_with_tools.invoke(messages + tool_messages)
        answer = final_response.content
    else:
        answer = response.content

    if isinstance(answer, list):
        answer = " ".join([str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in answer])
    elif isinstance(answer, dict):
        answer = answer.get("text", str(answer))

    return make_safe(str(answer), [question])