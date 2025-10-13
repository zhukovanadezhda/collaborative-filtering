import numpy as np
import logging

logger = logging.getLogger(__name__)

def build_item_similarity_from_genres(genres: np.ndarray, topk: int=50, eps: float=1e-8) -> np.ndarray:
    # row-normalize
    g = genres / np.maximum(genres.sum(axis=1, keepdims=True), 1.0)
    # cosine similarity (n x n)
    norms = np.sqrt((g * g).sum(axis=1, keepdims=True)) + eps
    g_norm = g / norms
    S = g_norm @ g_norm.T
    np.fill_diagonal(S, 0.0)
    # keep topk per row
    n = S.shape[0]
    for i in range(n):
        if topk < n:
            idx = np.argpartition(S[i], -(topk))[:-(topk)]
            S[i, idx] = 0.0
    # symmetrize (max to keep strongest)
    S = np.maximum(S, S.T)
    return S