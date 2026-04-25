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

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None  # loaded once and reused


def _get_model():
    global _model
    if _model is None:
        print(f"[Embedder] Loading model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts):
    """
    Embed a list of strings.
    Returns numpy array of shape (N, 384), normalised.
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalise for cosine similarity
    )
    return embeddings.astype(np.float32)


def embed_query(query):
    """
    Embed a single query string.
    Returns numpy array of shape (1, 384), normalised.
    """
    model = _get_model()
    vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vec.astype(np.float32)
