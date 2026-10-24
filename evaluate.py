import os
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()

# Replace with your actual Gemini API key or load it securely
api_key = os.environ.get("GOOGLE_API_KEY")

# Initialize Gemini model with the explicit key parameter
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=api_key, temperature=0.0)

question = "Is Apple a good buy right now?"
answer = "Regarding your interest in Apple (AAPL), the stock is currently trading at $348.22, reflecting a 3.3% increase over the past month."
context = "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories."

prompt = f"""
You are an AI evaluation judge. Evaluate the following answer based on the provided context and question. Give a score between 0.0 and 1.0 for Faithfulness and Answer Relevancy.

Question: {question}
Context: {context}
Answer: {answer}

Provide your evaluation in format:
Faithfulness: [Score]
Relevancy: [Score]
Reasoning: [Brief explanation]
"""

response = llm.invoke(prompt)
print("=== Evaluation Metrics (Gemini Judge) ===")
print(response.content[0]["text"])