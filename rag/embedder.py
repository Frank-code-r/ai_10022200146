# embedder.py
# Author: Frank Afelete Kofi Dogli | Index: 10022200146
# Part B: Embedding Pipeline

"""
EMBEDDING DESIGN DECISIONS
===========================
Model: all-MiniLM-L6-v2 (sentence-transformers)
- Produces 384-dimensional vectors
- Runs locally — no API needed for embeddings
- Fast and accurate for semantic similarity
- Free and open source

All vectors are L2-normalised so that cosine similarity
equals dot product — this lets FAISS do exact cosine search.
"""

# embedder.py
# Author: [Your Name] | Index: [Your Index Number]
# Part B: Embedding Pipeline — TF-IDF vectors (fast, no GPU needed)

import numpy as np
import re
from collections import defaultdict
import math
import pickle
import os

STORE_DIR = "data/vector_store"
os.makedirs(STORE_DIR, exist_ok=True)
IDF_PATH = os.path.join(STORE_DIR, "idf.pkl")

_idf = {}
_vocab = {}


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def build_tfidf_index(texts):
    """Build IDF table and vocabulary from all texts."""
    global _idf, _vocab
    N = len(texts)
    df = defaultdict(int)
    all_terms = set()

    for text in texts:
        terms = set(_tokenize(text))
        for t in terms:
            df[t] += 1
        all_terms.update(terms)

    _vocab = {t: i for i, t in enumerate(sorted(all_terms))}
    _idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in _vocab}

    # Save IDF
    with open(IDF_PATH, "wb") as f:
        pickle.dump((_idf, _vocab), f)

    print(f"[Embedder] TF-IDF index built: {len(_vocab)} terms")


def load_tfidf_index():
    global _idf, _vocab
    if os.path.exists(IDF_PATH):
        with open(IDF_PATH, "rb") as f:
            _idf, _vocab = pickle.load(f)
        return True
    return False


def _text_to_vec(text):
    tokens = _tokenize(text)
    total = len(tokens) if tokens else 1
    tf = defaultdict(int)
    for t in tokens:
        tf[t] += 1

    vec = np.zeros(len(_vocab), dtype=np.float32)
    for t, count in tf.items():
        if t in _vocab:
            vec[_vocab[t]] = (count / total) * _idf.get(t, 1.0)

    # L2 normalise
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def embed_texts(texts):
    """Convert list of texts to TF-IDF vectors."""
    build_tfidf_index(texts)
    print(f"[Embedder] Embedding {len(texts)} texts...")
    vecs = np.array([_text_to_vec(t) for t in texts], dtype=np.float32)
    print(f"[Embedder] Done. Shape: {vecs.shape}")
    return vecs


def embed_query(query):
    """Convert a single query to a TF-IDF vector."""
    if not _vocab:
        load_tfidf_index()
    vec = _text_to_vec(query)
    return vec.reshape(1, -1)
