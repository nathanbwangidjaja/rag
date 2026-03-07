import json
import os
import numpy as np
from pathlib import Path
from app.config import DATA_DIR

STORE_VECTORS_FILE = os.path.join(DATA_DIR, "vectors.npy")
STORE_META_FILE = os.path.join(DATA_DIR, "metadata.json")


class VectorStore:
    """Numpy-backed in-memory store. Saves to .npy + .json on disk."""

    def __init__(self):
        self.vectors: np.ndarray | None = None  # shape: (n, dim)
        self.chunks: list[dict] = []  # parallel list of chunk metadata
        self._load()

    def add(self, chunks: list[dict], embeddings: list[list[float]]):
        """Add chunks with their embeddings to the store."""
        new_vectors = np.array(embeddings, dtype=np.float32)

        if self.vectors is None:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])

        self.chunks.extend(chunks)
        self._save()

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Top-k cosine similarity search."""
        if self.vectors is None or len(self.chunks) == 0:
            return []

        qvec = np.array(query_embedding, dtype=np.float32)

        # cosine similarity: dot(a, b) / (||a|| * ||b||)
        norms = np.linalg.norm(self.vectors, axis=1)
        qnorm = np.linalg.norm(qvec)

        # avoid division by zero for empty/zero vectors
        safe_norms = np.where(norms == 0, 1.0, norms)
        safe_qnorm = qnorm if qnorm > 0 else 1.0

        similarities = self.vectors @ qvec / (safe_norms * safe_qnorm)

        # get top_k indices (argsort ascending, flip for descending)
        k = min(top_k, len(self.chunks))
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(similarities[idx])
            results.append(chunk)

        return results

    def count(self) -> int:
        return len(self.chunks)

    def clear(self):
        """Wipe everything."""
        self.vectors = None
        self.chunks = []
        self._save()

    def _save(self):
        os.makedirs(DATA_DIR, exist_ok=True)

        if self.vectors is not None:
            np.save(STORE_VECTORS_FILE, self.vectors)

        with open(STORE_META_FILE, "w") as f:
            json.dump(self.chunks, f)

    def _load(self):
        if os.path.exists(STORE_VECTORS_FILE) and os.path.exists(STORE_META_FILE):
            self.vectors = np.load(STORE_VECTORS_FILE)
            with open(STORE_META_FILE, "r") as f:
                self.chunks = json.load(f)
        else:
            self.vectors = None
            self.chunks = []


store = VectorStore()
