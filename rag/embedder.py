"""
embedder.py — Turn text chunks into dense float vectors using sentence-transformers.

What is an embedding?
    A dense vector (e.g. 384 floats) that encodes the *meaning* of a sentence.
    Similar sentences -> nearby vectors in 384-dimensional space.

    "The king ruled the land"    -> [0.12, -0.34, 0.89, ...]
    "A monarch governed his realm" -> [0.11, -0.31, 0.91, ...]  <- close!
    "I like pizza"               -> [-0.67, 0.22, -0.11, ...]  <- far away

    This is what makes semantic search work — finding chunks that mean the
    same thing as the query, even if they share no keywords.

Model: all-MiniLM-L6-v2
    - 384-dimensional output
    - 22M parameters (very small, fast on CPU)
    - Trained on 1B+ sentence pairs for semantic similarity
    - Downloads once (~90MB) and caches in ~/.cache/huggingface/
"""

import numpy as np
from sentence_transformers import SentenceTransformer # type: ignore


# Default model — best CPU-friendly balance of speed and quality
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """
    Thin wrapper around SentenceTransformer that:
      - Downloads and caches the model on first use
      - Returns L2-normalised float32 numpy arrays
        (normalised so dot product == cosine similarity)
      - Supports batch encoding for efficiency
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        print(f"Loading embedding model: {model_name}")
        print("  (downloads ~90MB on first run, cached after that)")
        self.model = SentenceTransformer(model_name)
        # Support both old and new API name
        if hasattr(self.model, 'get_embedding_dimension'):
            self.dim = self.model.get_embedding_dimension()
        else:
            self.dim = self.model.get_sentence_embedding_dimension()
        print(f"  Embedding dim : {self.dim}")

    def embed(self, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        """
        Encode a list of strings into a (N, dim) float32 array.

        Normalised so that:
            cosine_similarity(a, b) == np.dot(a, b)   (since ||a|| = ||b|| = 1)

        Args:
            texts:         List of strings to embed.
            batch_size:    How many to encode at once (trade RAM for speed).
            show_progress: Print a progress bar for large batches.

        Returns:
            np.ndarray of shape (len(texts), self.dim), dtype float32.
        """
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # L2 normalise -> cosine = dot product
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string. Returns shape (dim,)."""
        return self.embed([text])[0]


# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    embedder = Embedder()

    sentences = [
        "The king ruled his kingdom with wisdom.",
        "A monarch governed his realm with intelligence.",
        "I enjoy eating pasta for dinner.",
        "Python is a great programming language.",
    ]

    print("\nEmbedding sentences ...")
    vecs = embedder.embed(sentences)
    print(f"Matrix shape: {vecs.shape}  (4 sentences x {embedder.dim} dims)")

    # Cosine similarity matrix (dot product since normalised)
    sim = vecs @ vecs.T
    print("\nCosine similarity matrix:")
    print("              king    monarch   pasta   python")
    for i, s in enumerate(sentences):
        label = s[:12].ljust(12)
        row = "  ".join(f"{sim[i, j]:.3f}" for j in range(len(sentences)))
        print(f"  {label}  {row}")

    print("\nExpected: king<->monarch ≈ 0.9, king<->pasta ≈ 0.1")
