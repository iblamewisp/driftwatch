"""
Embedding similarity utilities.

Pure functions — no I/O, no model calls.
"""

import numpy as np


def mean_cosine_similarity(embedding: list[float], golden_embeddings: list[list[float]]) -> float:
    """
    Batch cosine similarity between one vector and a set of golden embeddings.
    Returns the mean similarity score across all goldens.
    """
    vec = np.array(embedding, dtype=np.float32)
    mat = np.array(golden_embeddings, dtype=np.float32)  # (N, D)
    vec_norm = np.linalg.norm(vec)
    mat_norms = np.linalg.norm(mat, axis=1)
    denom = mat_norms * vec_norm
    mask = denom > 0
    sims = np.where(mask, np.dot(mat, vec) / np.where(mask, denom, 1.0), 0.0)
    return float(np.mean(sims))
