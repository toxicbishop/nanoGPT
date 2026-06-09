"""
store.py — FAISS vector store: add, search, save, load.

What is FAISS?
    Facebook AI Similarity Search — a library for fast nearest-neighbour
    search over large sets of dense vectors.

    Given a query vector q and a database of N vectors,
    find the k vectors closest to q (by L2 distance or inner product).

Why not just numpy?
    Brute-force numpy dot product: O(N * dim) per query.
    Fine for N < 10K, but FAISS uses:
      - SIMD/AVX instructions for fast CPU arithmetic
      - Index structures (IVF, HNSW) for sub-linear search at scale
    We use IndexFlatIP here (exact brute-force inner product) — simple and
    correct for learning purposes. Easy to swap for HNSW later.

Index type used: IndexFlatIP
    IP = Inner Product.
    Since our embeddings are L2-normalised, inner product == cosine similarity.
    Higher score = more similar (range -1 to 1).
"""

import os
import json
import pickle

import numpy as np
import faiss # pyright: ignore[reportMissingImports]


class VectorStore:
    """
    Simple FAISS-backed vector store that stores:
      - An index (the embedding matrix + search structure)
      - A list of original text chunks (so we can return text, not just IDs)
      - Optional metadata per chunk (source file, page number, etc.)

    Usage:
        store = VectorStore(dim=384)
        store.add(embeddings, chunks)       # add numpy array + text list
        results = store.search(query_vec, k=5)  # returns list of (score, text)
        store.save("my_index")
        store = VectorStore.load("my_index")
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: Embedding dimension (must match your Embedder's output).
        """
        self.dim = dim
        # IndexFlatIP: exact inner product search (= cosine when normalised)
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[str] = []          # raw text for each stored vector
        self.metadata: list[dict] = []       # optional per-chunk metadata

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        return self.index.ntotal

    def add(
        self,
        embeddings: np.ndarray,   # (N, dim) float32
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        """
        Add a batch of embeddings and their corresponding text chunks.

        Args:
            embeddings: float32 array of shape (N, dim).
            texts:      List of N raw text strings.
            metadata:   Optional list of N dicts (e.g. {"source": "file.txt"}).
        """
        assert embeddings.shape[0] == len(texts), (
            f"embeddings ({embeddings.shape[0]}) and texts ({len(texts)}) must match"
        )
        assert embeddings.dtype == np.float32, "FAISS requires float32"

        self.index.add(embeddings)
        self.chunks.extend(texts)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(texts))

        print(f"  Added {len(texts)} chunks | Total: {self.size}")

    def search(
        self,
        query: np.ndarray,   # (dim,) or (1, dim) float32
        k: int = 5,
    ) -> list[tuple[float, str, dict]]:
        """
        Find the top-k most similar chunks to a query vector.

        Returns:
            List of (score, text, metadata) tuples, sorted by score descending.
            score = cosine similarity (0 to 1 for normalised vectors).
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)   # FAISS needs (1, dim)

        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:   # FAISS returns -1 for missing results
                continue
            results.append((float(score), self.chunks[idx], self.metadata[idx]))

        return results   # already sorted by score desc

    def save(self, directory: str) -> None:
        """
        Persist the index and chunk texts to disk.

        Saves:
            <directory>/index.faiss   — the FAISS binary index
            <directory>/chunks.pkl    — chunk texts + metadata
            <directory>/meta.json     — store metadata (dim, size)
        """
        os.makedirs(directory, exist_ok=True)

        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))

        with open(os.path.join(directory, "chunks.pkl"), "wb") as f:
            pickle.dump({"chunks": self.chunks, "metadata": self.metadata}, f)

        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump({"dim": self.dim, "size": self.size}, f)

        print(f"VectorStore saved -> {directory}/ ({self.size} vectors, dim={self.dim})")

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        """Load a previously saved VectorStore."""
        with open(os.path.join(directory, "meta.json")) as f:
            meta = json.load(f)

        store = cls(dim=meta["dim"])
        store.index = faiss.read_index(os.path.join(directory, "index.faiss"))

        with open(os.path.join(directory, "chunks.pkl"), "rb") as f:
            data = pickle.load(f)
        store.chunks = data["chunks"]
        store.metadata = data["metadata"]

        print(f"VectorStore loaded <- {directory}/ ({store.size} vectors, dim={store.dim})")
        return store


# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    dim = 8   # tiny fake dimension for demo
    store = VectorStore(dim=dim)

    # Random normalised fake vectors
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((5, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    texts = [
        "The king ruled with wisdom.",
        "A dragon guards the treasure.",
        "Python lists are mutable sequences.",
        "Attention is the core of transformers.",
        "The queen wore a golden crown.",
    ]

    store.add(vecs, texts)
    print(f"\nStore size: {store.size}")

    # Query with the first vector (should return itself as top hit)
    results = store.search(vecs[0], k=3)
    print("\nTop-3 similar to sentence 0:")
    for score, text, _ in results:
        print(f"  {score:.4f}  |  {text}")

    # Save and reload
    with tempfile.TemporaryDirectory() as tmpdir:
        store.save(tmpdir)
        store2 = VectorStore.load(tmpdir)
        results2 = store2.search(vecs[0], k=3)
        assert results == results2
        print("\nSave/load round-trip: OK")
