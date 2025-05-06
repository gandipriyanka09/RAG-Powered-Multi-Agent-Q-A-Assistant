# RAG-Powered Multi-Agent Q&A Assistant

## Overview
This is a simple multi-agent assistant that uses Retrieval-Augmented Generation (RAG) and basic agent logic to answer queries intelligently.

## Features
- ✅ Retrieval from a small document set (RAG via FAISS)
- ✅ LLM-based question answering (OpenAI)
- ✅ Agentic decision-making (LangChain agent routing)
- ✅ Minimal UI using Streamlit

## How It Works
1. **If the query contains 'calculate' or 'define'**, it's routed to a tool:
   - `calculate` → basic math evaluator
   - `define` → dictionary API
2. **Otherwise**, it performs:
   - Similarity search with FAISS
   - Answer generation using OpenAI GPT

## Installation
```bash
pip install -r requirements.txt
```

## Run
```bash
streamlit run app/app.py
```

## Folder Structure
```
rag_multi_agent_assistant/
├── app/
│   └── app.py
├── docs/
│   └── doc1.txt
│   └── doc2.txt
│   └── doc3.txt
├── README.md
└── requirements.txt
```