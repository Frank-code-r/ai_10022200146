# pipeline.py
# Author: Frank Afelete Kofi Dogli | Index: 10022200146
# Part D: Full RAG Pipeline

import requests
from rag.data_loader import load_csv, load_pdf, csv_records_to_text, pdf_pages_to_text
from rag.chunker import build_all_chunks
from rag.embedder import embed_texts
from rag.vector_store import VectorStore
from rag.retriever import HybridRetriever
from rag.prompt_builder import build_prompt
from rag.logger import log_stage, clear_session_log

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"


def _call_groq(system_prompt, user_message, api_key):
    """Call Groq API manually — no SDK."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }
    resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def build_index(force_rebuild=False):
    """
    Load data → chunk → embed → build FAISS index.
    Loads from disk if already built, unless force_rebuild=True.
    """
    store = VectorStore(dim=384)

    if not force_rebuild and store.load():
        retriever = HybridRetriever(store)
        retriever.build_index()
        return store, retriever

    print("[Pipeline] Building index from scratch...")

    # Stage 1: Load
    csv_records = load_csv()
    pdf_pages = load_pdf()
    csv_texts = csv_records_to_text(csv_records)
    pdf_texts = pdf_pages_to_text(pdf_pages)

    # Stage 2: Chunk
    all_chunks = build_all_chunks(csv_texts, pdf_texts)

    # Stage 3: Embed
    embeddings = embed_texts([c.text for c in all_chunks])

    # Stage 4: Store
    store.build(all_chunks, embeddings)
    store.save()

    # Stage 5: Retriever
    retriever = HybridRetriever(store)
    retriever.build_index()

    log_stage("index_built", {
        "total_chunks": len(all_chunks),
        "csv_chunks": sum(1 for c in all_chunks if c.source == "csv"),
        "pdf_chunks": sum(1 for c in all_chunks if c.source == "pdf"),
    })

    return store, retriever


def run_query(query, retriever, api_key, top_k=5, use_hybrid=True):
    """
    Full RAG pipeline for a single query:
    Query → Retrieve → Prompt → LLM → Response

    Logs every stage.
    """
    clear_session_log()

    # Stage 1: Receive query
    log_stage("query_received", {
        "query": query,
        "top_k": top_k,
        "mode": "hybrid" if use_hybrid else "vector_only",
    })

    # Stage 2: Retrieve
    if use_hybrid:
        retrieved = retriever.retrieve(query, top_k=top_k)
    else:
        retrieved = retriever.retrieve_vector_only(query, top_k=top_k)

    log_stage("retrieval_done", {
        "chunks_retrieved": len(retrieved),
        "scores": [round(s, 5) for _, s in retrieved],
        "sources": [c.source for c, _ in retrieved],
    })

    # Stage 3: Build prompt
    system_prompt, user_message, selected_chunks = build_prompt(query, retrieved)

    log_stage("prompt_built", {
        "chunks_selected": len(selected_chunks),
        "prompt_length_chars": len(user_message),
    })

    # Stage 4: Call LLM
    try:
        answer = _call_groq(system_prompt, user_message, api_key)
    except Exception as e:
        answer = f"LLM call failed: {e}"

    log_stage("llm_response", {
        "model": GROQ_MODEL,
        "answer_preview": answer[:200],
    })

    return {
        "answer": answer,
        "retrieved_chunks": retrieved,
        "selected_chunks": selected_chunks,
        "system_prompt": system_prompt,
        "user_message": user_message,
    }