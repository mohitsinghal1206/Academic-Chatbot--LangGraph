Academic Chatbot - LangGraph

A conditional RAG (Retrieval-Augmented Generation) chatbot built with LangGraph, designed to answer college students' questions about academics, fees, and general queries — personalized by their programme (BCA / BBA / B.Com H).

How it works

The app uses a LangGraph state graph with conditional routing:

Classifier node — an LLM call classifies the incoming query as academic, fee, or general.
Conditional routing — based on the classification:
academic → retrieves context from academics_handbook.pdf via a FAISS retriever
fee → retrieves context from fee_structure.pdf via a FAISS retriever
general → skips retrieval, answers from the LLM's own knowledge
Response node — generates a final answer personalized to the student's programme, using retrieved context when available.
Tech Stack
LangGraph — state graph orchestration
LangChain — document loading, splitting, retrieval
Groq (openai/gpt-oss-20b) — LLM inference
HuggingFace sentence-transformers — embeddings
FAISS — vector store
Streamlit — chat UI
