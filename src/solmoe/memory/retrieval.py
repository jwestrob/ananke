from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # pragma: no cover - optional
    _HAS_FAISS = False

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:  # pragma: no cover - optional
    NearestNeighbors = None  # type: ignore


class MotifMemory:
    """
    Build/query motif memory over embeddings; FAISS if available, fallback to sklearn NN.
    """

    def __init__(self, dim: int, use_faiss: bool = True):
        self.dim = dim
        self.use_faiss = use_faiss and _HAS_FAISS
        self.index = None
        self.meta: List[Dict[str, Any]] = []

    def build(self, embeddings: np.ndarray, meta: List[Dict[str, Any]]):
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError("embeddings must be [N,dim]")
        self.meta = list(meta)
        if self.use_faiss:
            self.index = faiss.IndexFlatL2(self.dim)
            self.index.add(embeddings.astype(np.float32))
        else:
            if NearestNeighbors is None:
                raise RuntimeError("Neither FAISS nor sklearn NearestNeighbors available")
            nn = NearestNeighbors(n_neighbors=20, metric="euclidean")
            nn.fit(embeddings)
            self.index = nn

    def query(self, embedding: np.ndarray, k: int = 20) -> List[Tuple[int, float, Dict[str, Any]]]:
        if self.index is None:
            raise RuntimeError("MotifMemory not built")
        if embedding.ndim != 1 or embedding.shape[0] != self.dim:
            raise ValueError("embedding must be [dim]")
        vec = embedding.reshape(1, -1).astype(np.float32)
        if self.use_faiss:
            D, I = self.index.search(vec, k)
            res = [(int(i), float(d), self.meta[int(i)]) for i, d in zip(I[0], D[0])]
        else:
            dists, idx = self.index.kneighbors(vec, n_neighbors=k)
            res = [(int(i), float(d), self.meta[int(i)]) for i, d in zip(idx[0], dists[0])]
        return res

