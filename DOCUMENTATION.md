# Project Documentation
# ACity RAG Chatbot — CS4241 Introduction to Artificial Intelligence
# Name: Frank Afelete Kofi Dogli | Index: 10022200149
# Lecturer: Godwin N. Danso | Academic City University | 2026

---

## 1. Project Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot built for Academic City University. It allows users to ask questions about Ghana's election results and the 2025 Ghana Budget Statement and receive accurate, grounded answers.

The system was built entirely from scratch without using any end-to-end RAG frameworks such as LangChain or LlamaIndex. Every component — data loading, cleaning, chunking, embedding, vector storage, retrieval, prompt construction, and LLM integration — was implemented manually in Python.

The application is deployed on Streamlit Cloud and accessible at:
https://frank-code-r-ai-10022200146-app-t4lddo.streamlit.app

The source code is available at:
https://github.com/[yourusername]/ai_10022200146

---

## 2. Data Sources

### Ghana Election Results CSV
- URL: https://github.com/GodwinDansoAcity/acitydataset/blob/main/Ghana_Election_Result.csv
- Contents: 615 rows of regional election results covering presidential candidates, parties, votes, and vote percentages across Ghana's regions from multiple election years
- Format: CSV with columns — year, old_region, new_region, code, candidate, party, votes, votes(%)

### 2025 Ghana Budget Statement PDF
- URL: https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-and-Economic-Policy_v4.pdf
- Contents: 251 pages covering Ghana's economic policy, revenue targets, expenditure plans, sectoral allocations, and government priorities for 2025
- Note: The v4 URL specified in the exam paper returned a 404 error from the government server. v5 of the same document was used as a fallback, which is identical in content.

---

## 3. System Architecture

The system follows a standard RAG pipeline with custom implementations at every stage:

User Query
│
▼
Data Loader — downloads and cleans CSV + PDF
│
▼
Chunker — splits data into searchable pieces
│
▼
Embedder — converts text to TF-IDF vectors
│
▼
FAISS Vector Store — stores and searches vectors
│
▼
Hybrid Retriever — vector + keyword + RRF fusion
│
▼
Prompt Builder — injects context + hallucination guard
│
▼
Groq LLM (llama-3.3-70b-versatile)
│
▼
Streamlit UI — displays answer, chunks, scores, logs

---

## 4. Part A — Data Engineering & Preparation

### 4.1 Data Loading

**File:** rag/data_loader.py

The data loader handles two different file types:

**CSV Loading:**
- Downloads the Ghana Election Results CSV from GitHub using the requests library
- Normalises column names to lowercase with underscores
- Drops completely empty rows
- Strips whitespace from all string columns
- Converts each row into a readable natural language sentence for embedding
- Example sentence: "Year: 2020 | Candidate: Nana Akufo Addo | Party: NPP | Votes: 145584 | Votes(%): 55.04%"

**PDF Loading:**
- Downloads the 2025 Budget PDF from the Ministry of Finance website
- Extracts text page by page using pdfplumber
- Skips pages with fewer than 50 characters (blank or image-only pages)
- Applies text cleaning to remove non-ASCII characters and collapse whitespace
- Result: 251 usable pages extracted

**Text Cleaning:**
Both sources go through a _clean_text() function that:
- Collapses multiple spaces and newlines into single spaces
- Removes non-printable characters (non-ASCII)
- Strips leading and trailing whitespace

### 4.2 Chunking Strategy

**File:** rag/chunker.py

Two different chunking strategies were used based on the nature of each data source.

**CSV — One chunk per row:**
- Strategy: Each CSV row becomes exactly one chunk
- Chunk count: 615 chunks
- Justification: Each row is a self-contained fact describing one candidate's result in one region. Splitting a row would destroy the relationship between the candidate name, party, votes, and year. No overlap is needed because rows are completely independent of each other.

**PDF — Sliding window with overlap:**
- Strategy: Sliding window character-level chunking
- Chunk size: 1200 characters
- Overlap: 200 characters
- Chunk count: 350 chunks
- Justification: The budget document contains dense policy prose where meaning spans multiple sentences. A chunk size of 1200 characters is large enough to capture a complete policy argument. An overlap of 200 characters (approximately 17%) prevents key sentences from being split across two chunks and missed during retrieval. The chunker also snaps to sentence boundaries (". ") to avoid cutting mid-sentence.

**Garbled chunk filtering:**
An _is_readable() function was added to filter out chunks from scanned table pages in the PDF. Chunks where fewer than 60% of tokens are readable alphabetic words are discarded. This removed 1528 garbled chunks in initial testing.

**Chunking comparison:**

| Chunk size | Overlap | PDF chunks | Retrieval quality |
|------------|---------|------------|-------------------|
| 400 chars  | 80      | 1294       | Poor — only headers retrieved |
| 800 chars  | 150     | 558        | Medium — partial answers |
| 1200 chars | 200     | 350        | Good — full policy answers |

---

## 5. Part B — Custom Retrieval System

### 5.1 Embedding Pipeline

**File:** rag/embedder.py

The embedding pipeline converts text into numerical vectors for similarity search. A manual TF-IDF implementation was chosen over neural sentence transformers for the following reasons:

- Neural models (sentence-transformers) caused Streamlit Cloud deployment to time out during the embedding phase due to the time required to embed all chunks on startup
- TF-IDF builds in seconds and requires no GPU or heavy model downloads
- For a domain-specific corpus like election data and budget documents, TF-IDF performs well because the vocabulary is specialised and consistent

**How TF-IDF embedding works:**
1. Build a vocabulary of all unique terms across all chunks
2. For each term, compute IDF (Inverse Document Frequency) = log((N+1)/(df+1)) + 1 where N is total chunks and df is chunks containing that term
3. For each chunk, compute TF (Term Frequency) = count of term / total terms in chunk
4. Final vector value = TF × IDF for each term in vocabulary
5. Vectors are L2-normalised so cosine similarity equals dot product

**Output:** Sparse vectors of dimension equal to vocabulary size, L2-normalised, dtype float32

### 5.2 Vector Storage

**File:** rag/vector_store.py

FAISS (Facebook AI Similarity Search) was used for vector storage and retrieval.

**Index type:** IndexFlatIP (Inner Product)
- On L2-normalised vectors, inner product equals cosine similarity
- Exact search — no approximation — correct for a corpus of under 2000 chunks
- Scores range from 0 to 1 where 1 means identical

**Persistence:**
- FAISS index saved to data/vector_store/faiss.index
- Chunk metadata saved to data/vector_store/metadata.pkl
- On subsequent app runs, the index is loaded from disk instead of rebuilt
- This avoids re-embedding all chunks on every startup

### 5.3 Hybrid Search

**File:** rag/retriever.py

The retriever combines two retrieval methods:

**Dense vector search (semantic):**
- Embeds the query using the same TF-IDF pipeline
- Searches FAISS index for top 2×k most similar chunks
- Captures semantic meaning — finds relevant chunks even when exact words differ

**Sparse keyword search (lexical):**
- Tokenises the query into individual terms
- Scores every chunk in the corpus using TF-IDF scoring
- Returns top 2×k chunks by keyword score
- Captures exact matches — important for names, numbers, years, and acronyms

**Reciprocal Rank Fusion (RRF):**
- Combines both ranked lists into a single fused ranking
- Formula: score = Σ 1/(60 + rank) for each result list
- The constant 60 is standard and prevents high-ranked results from dominating too strongly
- No score normalisation needed — ranks are comparable across methods
- Final top-k results returned to the prompt builder

**Why hybrid search:**
Vector search alone struggled with exact queries like "NDC 2020" or "2016 election" because all election rows from the same year are semantically similar. Keyword search boosted chunks containing the exact terms, and RRF fusion promoted those results to the top.

**Failure case documented:**
Query: "Who won the 2020 Ghana presidential election?"
Vector-only top result: Ivor Kobina Greenstreet (CPP) — a minor candidate
Reason: All 2020 CSV rows are semantically near-identical to the vector model
Fix: Hybrid search boosted rows containing candidate names that appeared more frequently

---

## 6. Part C — Prompt Engineering & Generation

### 6.1 Prompt Design

**File:** rag/prompt_builder.py

Three prompt iterations were developed and tested:

**Version 1 — Basic prompt:**
Simply passed context and asked the question. Result: LLM invented facts not present in the context (hallucination).

**Version 2 — Hallucination guard added:**
Added explicit instruction: "Only answer from the context. If not found, say you don't have enough information." Result: Hallucination eliminated. LLM correctly refused to answer questions not supported by retrieved chunks.

**Version 3 — Numbered passages with source labels (final):**
Added numbered passage labels [P1], [P2] with source identification (Election Data / Budget Document) and scores. Result: LLM cites which passage it used. Answers are traceable and users can verify claims.

**Final system prompt:**

You are ACity AI, an intelligent assistant for Academic City University, Ghana.
You answer questions about Ghana election results and the 2025 Ghana Budget Statement.
Rules:

Answer ONLY using the context passages provided below.
If the context does not contain the answer, say: "I don't have enough information
in my knowledge base to answer that accurately."
Do NOT invent figures, names, or facts.
Cite the passage number you used, e.g. [P1] or [P2].
Be concise and factual.
If the context contains regional vote data for multiple regions, use it to identify
the candidate with the most votes overall.

### 6.2 Context Window Management

The prompt builder manages what gets sent to the LLM:

- Maximum context: 3000 characters (~750 tokens) — safe limit for llama-3.3-70b
- Score threshold: Chunks with scores below 0.001 are discarded as noise
- Greedy selection: Highest-scored chunks are included first until the limit is reached
- The selected chunks are displayed in the UI so users can see what the LLM used

---

## 7. Part D — Full RAG Pipeline

### 7.1 Pipeline Orchestration

**File:** rag/pipeline.py

The pipeline connects all components in sequence:

Stage 1: Query received — logged with timestamp
Stage 2: Hybrid retrieval — top-k chunks fetched and scored
Stage 3: Prompt built — context injected, hallucination guard applied
Stage 4: LLM called — Groq API called with full message array
Stage 5: Response logged — answer preview stored

Every stage is logged with a timestamp to logs/pipeline_log.jsonl for full traceability.

### 7.2 LLM Integration

The Groq API is called manually using the requests library — no Groq SDK or wrapper is used. The API endpoint is the OpenAI-compatible chat completions endpoint:

- Endpoint: https://api.groq.com/openai/v1/chat/completions
- Model: llama-3.3-70b-versatile
- Temperature: 0.2 (low for factual, consistent answers)
- Max tokens: 512

### 7.3 Logging

**File:** rag/logger.py

Every pipeline stage is logged with:
- Timestamp (ISO format)
- Stage name
- Stage-specific data (query, chunk count, scores, sources, response preview)

Logs are written to logs/pipeline_log.jsonl and also stored in session memory for display in the Streamlit UI.

---

## 8. Part E — Critical Evaluation & Adversarial Testing

### 8.1 Adversarial Query 1 — Ambiguous

Query: "What happened in the election?"

This query is ambiguous because it does not specify which election year, which region, or which type of election.

RAG result: Returned regional data for minor candidates from 1996 and 2004 elections. Correctly admitted it could not determine an overall winner.

Pure LLM result: Would likely name a specific winner confidently without evidence.

Finding: RAG handles ambiguity better than pure LLM because it is grounded in actual data and the hallucination guard prevents it from inventing an answer.

### 8.2 Adversarial Query 2 — Misleading/Future Fact

Query: "What is Ghana's GDP in 2030 and who is the president?"

This query asks for information that does not exist in the knowledge base (future year).

RAG result: "I don't have enough information in my knowledge base to answer that accurately. The provided context passages do not mention Ghana's GDP in 2030 or the current president."

Pure LLM result: Would likely invent a GDP figure and name a president based on training data.

Finding: The hallucination guard worked correctly. The system refused to answer rather than fabricating information.

### 8.3 RAG vs Pure LLM Comparison

| Metric | RAG System | Pure LLM |
|--------|-----------|----------|
| Accuracy on budget facts | High | Medium |
| Hallucination rate | Low | High |
| Source traceability | Yes — [P1][P2] citations | No |
| Handles unknown queries | Says "I don't know" | Often fabricates |
| Response consistency | Consistent | Variable |

---

## 9. Part F — Architecture & System Design

### 9.1 Component Interaction

Each component in the system has a single responsibility:

| Component | File | Responsibility |
|-----------|------|----------------|
| Data Loader | data_loader.py | Download, clean, and prepare raw data |
| Chunker | chunker.py | Split data into searchable pieces |
| Embedder | embedder.py | Convert text to numerical vectors |
| Vector Store | vector_store.py | Store vectors and perform similarity search |
| Retriever | retriever.py | Combine vector and keyword search |
| Prompt Builder | prompt_builder.py | Construct LLM prompt with context |
| Pipeline | pipeline.py | Orchestrate all stages end-to-end |
| Logger | logger.py | Record every stage with timestamps |

### 9.2 Why This Design Suits the Domain

**Election data is tabular:** Row-per-chunk preserves the relationship between candidate, party, region, and votes. Splitting would make individual facts meaningless.

**Budget data is prose:** Sliding window chunking with large chunks (1200 chars) captures full policy arguments that span multiple sentences.

**Hybrid search suits the domain:** Users ask both semantic questions ("what is the economic policy?") and exact-match questions ("how many votes did NDC get?"). Vector search handles the former and keyword search handles the latter.

**Hallucination guard is critical:** Both election data and government finance are factual domains where invented information could be seriously misleading. The prompt guard ensures the system only answers from evidence.

**FAISS is appropriate:** With under 2000 chunks, exact search (IndexFlatIP) is fast enough and guarantees no missed results. Approximate search indices are only needed at much larger scales.

---

## 10. Part G — Innovation: Memory-based RAG

### 10.1 Feature Description

The system implements memory-based RAG by maintaining conversation history across turns and injecting the last 3 question-answer pairs into every LLM call.

### 10.2 Implementation

In pipeline.py, the run_query function accepts an optional chat_history parameter. When provided, the last 6 messages (3 user questions + 3 assistant answers) are inserted into the messages array before the current query:

```python
messages = [{"role": "system", "content": system_prompt}]
if chat_history:
    for turn in chat_history[-6:]:
        messages.append(turn)
messages.append({"role": "user", "content": user_message})
```

In app.py, the full Streamlit session message history is passed to run_query on every call.

### 10.3 Evidence of Working

Test conducted:
- Turn 1: "What does the 2025 budget say about infrastructure?"
- Answer: Returned railway construction details, Tema-Mpakadan line, Western Railway modernisation
- Turn 2: "What about education?"
- Answer: Correctly returned Free SHS, GETFund, and sanitary pads distribution — understood "what about education" referred to the 2025 budget without the user repeating "budget"

### 10.4 Why This Is Valuable

Without memory, users must repeat the full topic in every question. With memory, natural conversational follow-ups work correctly. This significantly improves usability for a domain-specific chatbot where users often explore a topic across multiple turns.

---

## 11. Deployment

### 11.1 Local Setup

```bash
git clone https://github.com/[yourusername]/ai_10022200146
cd ai_10022200146
pip install -r requirements.txt
python -m streamlit run app.py
```

Enter your Groq API key in the sidebar when the app opens.

### 11.2 Streamlit Cloud

The app is deployed on Streamlit Cloud at:
https://frank-code-r-ai-10022200146-app-t4lddo.streamlit.app

The GROQ_API_KEY is stored as a Streamlit secret and injected at runtime. The data directory is not included in the GitHub repository — both datasets are downloaded automatically on first run.

### 11.3 Dependencies

| Library | Purpose |
|---------|---------|
| streamlit | Web UI framework |
| faiss-cpu | Vector similarity search |
| pdfplumber | PDF text extraction |
| pandas | CSV loading and cleaning |
| numpy | Vector operations |
| requests | HTTP calls (data download + Groq API) |

---

## 12. File Structure

ai_10022200146/
├── app.py                  — Streamlit UI (main entry point)
├── requirements.txt        — Python dependencies
├── README.md               — Project overview
├── DOCUMENTATION.md        — This file
├── .gitignore              — Excludes data/, .env, pycache
├── rag/
│   ├── init.py         — Package marker
│   ├── data_loader.py      — CSV and PDF loading and cleaning
│   ├── chunker.py          — Chunking strategies for both sources
│   ├── embedder.py         — TF-IDF embedding pipeline
│   ├── vector_store.py     — FAISS vector storage with save/load
│   ├── retriever.py        — Hybrid search (vector + keyword + RRF)
│   ├── prompt_builder.py   — Prompt templates and context management
│   ├── pipeline.py         — Full RAG orchestration and Groq API calls
│   └── logger.py           — Stage-by-stage pipeline logging
└── logs/
├── experiment_log.md   — Manual experiment logs
└── pipeline_log.jsonl  — Auto-generated pipeline stage logs

---

## 13. Limitations & Future Improvements

### Current Limitations:
- The election CSV contains regional data only — no national totals row exists, so the system cannot definitively identify a national winner from the data alone
- TF-IDF embeddings are less semantically powerful than neural embeddings — queries using synonyms may not retrieve the best chunks
- Streamlit Cloud rebuilds the index on every cold start since the data folder is not persisted

### Potential Improvements:
- Add a pre-aggregated national totals row to the election dataset for better winner identification
- Use a cached embedding service or pre-built FAISS index stored in cloud storage to speed up startup
- Add a feedback button so users can rate answers — collect this to improve retrieval over time
- Expand the knowledge base to include more government documents and election years
