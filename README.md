# 📚 Enterprise Study Companion: Production-Ready RAG Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Integration-green)](https://python.langchain.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red)](https://qdrant.tech/)
[![MLflow](https://img.shields.io/badge/MLflow-Observability-blue)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud_Ready-FF4B4B)](https://streamlit.io/)

## 📌 Project Overview
The **Enterprise Study Companion** is an end-to-end, locally hosted Retrieval-Augmented Generation (RAG) application. It allows users to dynamically upload PDF textbooks, parses and vectorizes the knowledge base, and synthesizes accurate, citation-backed answers using an LLM.

Designed with **AI Engineering best practices**, this project intentionally avoids basic wrapper scripts in favor of a modular, object-oriented architecture. It features a persistent local vector database, ephemeral cloud-safe memory management, and an enterprise-grade observability layer for full token and latency tracking.

## 🏗️ System Architecture & Tech Stack
This pipeline is built to operate efficiently within constrained environments (e.g., 1GB RAM cloud containers) without sacrificing retrieval quality.

*   **Dependency Management:** `uv` (Ultra-fast Python package installer & resolver)
*   **Orchestration:** LangChain Expression Language (LCEL)
*   **Embedding Model:** `BAAI/bge-small-en-v1.5` (via FastEmbed for local, PyTorch-free execution)
*   **Vector Store:** Local Qdrant (SQLite backend, resolving cross-platform OS locking)
*   **LLM Engine:** gemini-3.6-flash
*   **Observability:** MLflow (Full nested tracing, prompt tracking, and API latency metrics)
*   **Frontend & Deployment:** Streamlit / Dockerized for CI/CD

## 🚀 Key Engineering Highlights
*   **Robust Ingestion Pipeline:** Custom `DocumentParser` and `IntelligentChunker` classes implementing recursive character splitting with calculated overlaps to prevent context boundary loss.
*   **Cloud-Native File Handling:** Built to survive ephemeral container environments. Implements Python's `tempfile` module to handle in-memory buffer writing and instant cleanup, preventing Out-Of-Memory (OOM) crashes and disk bloat.
*   **Advanced State Management:** Leverages Streamlit's caching decorators to split the database connection from the LangChain pipeline, allowing live vector insertion without triggering thread-lock collision errors.
*   **Transparent Observability:** Integrated `mlflow.langchain.autolog()` to record a "glass box" trace of every user query, providing instant visibility into exact chunks retrieved, system prompt formatting, and LLM generation time.

## 📂 Repository Structure

```text
Study-Companion/
├── scripts/                    # Developer testing & DB population tools
│   ├── ingest.py               
│   └── test_db.py              
├── src/
│   └── study_companion/
│       ├── __init__.py
│       ├── utils/              # Core application utilities
│       │   ├── logger.py       # Standardized logging configuration
│       │   └── exception.py    # Custom error classes and handlers
│       ├── ingestion/
│       │   ├── parser.py       # PyPDFLoader integration
│       │   └── chunker.py      # Overlap and token optimization
│       ├── retrieval/
│       │   ├── embedder.py     # FastEmbed configuration
│       │   └── vector_store.py # qdrant client management
│       └── generation/
│           ├── prompt.py       # Guardrails system prompts
│           └── chain.py        # LCEL logic
├── .env                        
├── .gitignore                  
├── app.py                      # Streamlit Frontend application
├── Dockerfile                  # Containerization instruction
└── requirements.txt            # Locked dependencies
```

## 💻 Local Installation & Setup
### 1. Clone the repository
```Bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd Study-Companion
```

### 2. Environment Setup (using uv)
Ensure uv is installed, then install dependencies:
```Bash
uv pip install -r requirements.txt
```

### 3. Configure API Keys
Create a .env file in the root directory and add your Google Gemini API key:
```
Code snippet
GEMINI_API_KEY="your_api_key_here"
```
### 4. Run the Application
```Bash
uv run streamlit run app.py
```
### 5. View MLflow Traces
To inspect retrieval latency, token usage, and LangChain nodes, launch the local MLflow tracking server:
```Bash
uv run mlflow ui --port 5003 --workers 1 --backend-store-uri sqlite:///mlflow.db
Navigate to http://127.0.0.1:5003 to view the Enterprise_Study_Companion experiment data.
```
### 🛣️ Future Roadmap
Agentic AI Integration: Migrating the linear LCEL chain to LangGraph to implement dynamic routing (e.g., routing web-search queries vs. local PDF queries).

Multi-Modal RAG: Extending the ingestion pipeline to parse diagrams and tables using Convolutional Neural Networks (CNNs).
