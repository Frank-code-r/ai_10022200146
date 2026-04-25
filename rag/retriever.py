# retriever.py
# Author: [Your Name] | Index: [Your Index Number]
# Part B: Hybrid Retrieval — Vector + Keyword + RRF Fusion

"""
HYBRID SEARCH DESIGN
=====================
Combines two retrieval methods:

1. Dense vector search (semantic):
   - Finds chunks by meaning, even if exact words differ
   - e.g. "fiscal policy" can match "government spending"

2. Sparse keyword search (lexical):
   - Finds chunks containing exact words from the query
   - Important for names, numbers, years, acronyms
   - e.g. "NDC 2020" must match exactly

Fusion: Reciprocal Rank Fusion (RRF)
   score = 1 / (60 + rank)  for each result list
   Then scores are summed and re-ranked.
   - No need to normalise scores across methods
   - Simple and effective

FAILURE CASE & FIX (documented in experiment log)
---------------------------------------------------
Problem: Vector-only search returned minor party candidates
when asked "Who won the 2020 election?" because all 2020
CSV rows are semantically similar.

Fix: Keyword search boosts chunks with exact terms like
"NPP" or "Akufo-Addo", and RRF fusion promotes them to top.
"""

import re
import math
from collections import defaultdict
from rag.vector_store import VectorStore
from rag.embedder import embed_query


# ── Keyword scoring ────────────────────────────────────────────────────────────

def _tokenize(text):
    """Split text into lowercase tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_idf(chunks):
    """Compute IDF score for every term across all chunks."""
    N = len(chunks)
    df = defaultdict(int)
    for chunk in chunks:
        for term in set(_tokenize(chunk.text)):
            df[term] += 1
    return {t: math.log((N + 1) / (freq + 1)) + 1 for t, freq in df.items()}


def _keyword_score(query_terms, chunk, idf):
    """Compute TF-IDF score for a chunk given query terms."""
    tokens = _tokenize(chunk.text)
    total = len(tokens) if tokens else 1
    tf = defaultdict(int)
    for t in tokens:
        tf[t] += 1
    return sum(
        (tf[t] / total) * idf.get(t, 1.0)
        for t in query_terms
        if t in tf
    )


# ── RRF Fusion ─────────────────────────────────────────────────────────────────

def _rrf_fuse(vector_results, keyword_results, k=60):
    """
    Combine two ranked lists using Reciprocal Rank Fusion.
    Returns merged list of (chunk, fused_score).
    """
    scores = defaultdict(float)
    chunk_map = {}

    for rank, (chunk, _) in enumerate(vector_results):
        scores[chunk.chunk_id] += 1 / (k + rank + 1)
        chunk_map[chunk.chunk_id] = chunk

    for rank, (chunk, _) in enumerate(keyword_results):
        scores[chunk.chunk_id] += 1 / (k + rank + 1)
        chunk_map[chunk.chunk_id] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(chunk_map[cid], score) for cid, score in ranked]


# ── Main Retriever ─────────────────────────────────────────────────────────────

class HybridRetriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self._idf = {}

    def build_index(self):
        """Pre-compute IDF table from all chunks."""
        self._idf = _build_idf(self.vector_store.chunks)
        print(f"[Retriever] IDF index built: {len(self._idf)} unique terms")

    def retrieve(self, query, top_k=5):
        """
        Hybrid retrieval: vector + keyword → RRF fusion.
        Returns list of (chunk, score) sorted by relevance.
        """
        # 1. Vector search
        q_vec = embed_query(query)
        vector_results = self.vector_store.search(q_vec, top_k=top_k * 2)

        # 2. Keyword search
        query_terms = _tokenize(query)
        keyword_scores = [
            (chunk, _keyword_score(query_terms, chunk, self._idf))
            for chunk in self.vector_store.chunks
        ]
        keyword_results = sorted(
            keyword_scores, key=lambda x: x[1], reverse=True
        )[: top_k * 2]

        # 3. Fuse and return top_k
        fused = _rrf_fuse(vector_results, keyword_results)[:top_k]
        return fused

    def retrieve_vector_only(self, query, top_k=5):
        """Vector-only retrieval — used for comparison in Part E."""
        q_vec = embed_query(query)
        return self.vector_store.search(q_vec, top_k=top_k)