# ACity RAG Chatbot — CS4241 Introduction to Artificial Intelligence

**Student Name:** Frank Afelete Kofi Dogli
**Index Number:** 10022200149
**Course:** CS4241 — Introduction to Artificial Intelligence
**Lecturer:** Godwin N. Danso
**Institution:** Academic City University
**Year:** 2026

---

## 🚀 Live Demo
https://frank-code-r-ai-10022200146-app-t4lddo.streamlit.app

## 📁 Repository
https://github.com/Frank-code-r/ai_10022200146

---

## 🧠 Project Overview

A fully custom Retrieval-Augmented Generation (RAG) chatbot built from scratch for Academic City University. The system answers questions about Ghana's election results and the 2025 Ghana Budget Statement.

Built without LangChain, LlamaIndex, or any pre-built RAG framework. All components — chunking, embedding, vector storage, retrieval, and prompt construction — were implemented manually.

---

## 📦 Data Sources

- Ghana Election Results CSV: https://github.com/GodwinDansoAcity/acitydataset/blob/main/Ghana_Election_Result.csv
- 2025 Ghana Budget Statement PDF: https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-and-Economic-Policy_v4.pdf

---

## 🏗️ Architecture

User Query
│
▼
[Data Loader] — downloads and cleans CSV + PDF
│
▼
[Chunker] — CSV: one chunk per row | PDF: sliding window 1200 chars, 200 overlap
│
▼
[Embedder] — TF-IDF vectors (manual implementation, no frameworks)
│
▼
[Vector Store] — FAISS IndexFlatIP (cosine similarity via dot product)
│
▼
[Hybrid Retriever] — Vector search + Keyword search → RRF Fusion
│
▼
[Prompt Builder] — numbered passages + hallucination guard
│
▼
[Groq LLM] — llama-3.3-70b-versatile
│
▼
[Streamlit UI] — answer + retrieved chunks + scores + pipeline log

---

## 🧩 Part-by-Part Implementation

### Part A — Data Engineering & Preparation
- CSV: downloaded and cleaned 615 rows, normalised column names, stripped whitespace
- PDF: extracted 251 pages using pdfplumber, filtered garbled/scanned pages using readability check
- CSV chunking: one chunk per row — each row is a self-contained election fact
- PDF chunking: sliding window 1200 characters, 200 character overlap, snaps to sentence boundaries
- Justification: CSV rows are independent facts so no overlap is needed. PDF needs overlap to prevent key sentences being cut at chunk boundaries

### Part B — Custom Retrieval System
- Manual TF-IDF embedding pipeline (no sentence-transformers, no OpenAI)
- FAISS IndexFlatIP vector store with save/load to disk
- Hybrid search: TF-IDF vector search + keyword scoring → Reciprocal Rank Fusion
- Failure case: vector-only search returned minor party candidates for election queries. Fixed by hybrid search boosting exact keyword matches

### Part C — Prompt Engineering
- 3 prompt iterations documented in experiment log
- v1: no hallucination guard — model invented facts
- v2: added "only answer from context" rule — hallucination eliminated
- v3: added numbered passages [P1][P2] with source labels — answers became traceable
- Context window management: max 3000 characters, greedy selection by score, score threshold filter

### Part D — Full RAG Pipeline
- pipeline.py orchestrates all stages: load → chunk → embed → retrieve → prompt → LLM → response
- logger.py logs every stage with timestamps
- UI displays retrieved chunks, similarity scores, final prompt, and pipeline log

### Part E — Adversarial Testing
- Query 1 (Ambiguous): "What happened in the election?" — system returned data but correctly admitted it could not determine a winner
- Query 2 (Misleading): "What is Ghana's GDP in 2030 and who is the president?" — system correctly refused to answer, no hallucination
- Hybrid RAG outperformed vector-only on both queries

### Part F — Architecture & System Design
- See architecture diagram above
- TF-IDF chosen over neural embeddings for cloud deployment speed
- FAISS chosen for exact cosine search on small corpus (under 2000 chunks)
- Hybrid search chosen to handle both semantic and exact-match queries

### Part G — Innovation
- Memory-based RAG: Streamlit session state preserves full conversation history
- Users can ask follow-up questions and the chat history is maintained across turns
- Pipeline logging: every stage logged with timestamp to logs/pipeline_log.jsonl for full traceability

---

## 🛠️ Local Setup

```bash
git clone https://github.com/[your-github-username]/ai_10022200146
cd ai_10022200146
pip install -r requirements.txt
python -m streamlit run app.py
```

Enter your Groq API key in the sidebar when the app opens.

---

## ☁️ Streamlit Cloud Deployment

1. Push code to GitHub
2. Go to share.streamlit.io
3. Connect repo → set app.py as entry point
4. Add GROQ_API_KEY in Settings → Secrets:
```toml
GROQ_API_KEY = "your_key_here"
```
5. Deploy

---

## 📋 File Structure

ai_10022200146/
├── app.py                  — Streamlit UI
├── requirements.txt        — Dependencies
├── README.md               — This file
├── .gitignore
├── rag/
│   ├── init.py
│   ├── data_loader.py      — CSV + PDF loading and cleaning
│   ├── chunker.py          — Manual chunking strategies
│   ├── embedder.py         — TF-IDF embedding pipeline
│   ├── vector_store.py     — FAISS vector storage
│   ├── retriever.py        — Hybrid search (vector + keyword + RRF)
│   ├── prompt_builder.py   — Prompt templates + context management
│   ├── pipeline.py         — Full RAG orchestration + Groq API calls
│   └── logger.py           — Stage-by-stage pipeline logging
└── logs/
└── experiment_log.md   — Manual experiment logs

---

## 📊 Stack

| Component | Technology |
|-----------|------------|
| UI | Streamlit |
| Embeddings | TF-IDF (manual implementation) |
| Vector Store | FAISS IndexFlatIP |
| Retrieval | Hybrid: TF-IDF vector + keyword + RRF |
| LLM | Groq — LLaMA-3.3-70B-Versatile |
| PDF parsing | pdfplumber |
| CSV parsing | pandas |
