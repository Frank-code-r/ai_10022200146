# vector_store.py
# Author: Frank Afelete Kofi Dogli | Index: 10022200146
# Part B: Vector Storage using FAISS

"""
VECTOR STORE DESIGN
====================
Index type: faiss.IndexFlatIP (Inner Product)
- On L2-normalised vectors, inner product = cosine similarity
- Scores range from 0 to 1 (1 = identical)
- Exact search — no approximation (fine for our ~3437 chunks)

Persistence: saves index + chunk metadata to disk
- Avoids re-embedding all 3437 chunks on every run
- Loads in seconds on subsequent runs
"""

import os
import pickle
import numpy as np
import faiss

STORE_DIR = "data/vector_store"
os.makedirs(STORE_DIR, exist_ok=True)

INDEX_PATH = os.path.join(STORE_DIR, "faiss.index")
META_PATH  = os.path.join(STORE_DIR, "metadata.pkl")


class VectorStore:
    def __init__(self, dim=None):
        self.dim = dim
        self.index = None
        self.chunks = []

    def build(self, chunks, embeddings):
        """Build FAISS index from chunks and their embeddings."""
        assert len(chunks) == embeddings.shape[0]
        self.chunks = chunks
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        print(f"[VectorStore] Built index with {self.index.ntotal} vectors")

    def save(self):
        """Save index and metadata to disk."""
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"[VectorStore] Saved to {STORE_DIR}")

    def load(self):
        if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
            return False
        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            self.chunks = pickle.load(f)
            from rag.embedder import load_tfidf_index
            load_tfidf_index()
            print(f"[VectorStore] Loaded {self.index.ntotal} vectors from disk")
            return True

    def search(self, query_vec, top_k=5):
        """
        Search for top_k most similar chunks.
        Returns list of (chunk, score) sorted by score descending.
        """
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results
