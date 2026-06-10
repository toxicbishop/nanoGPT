"""
store.py — ChromaDB vector store: add, search, save, load.

What is ChromaDB?
    A developer-focused open-source vector database designed to make it easy
    to build AI applications with embeddings.

    It handles:
      - Embedding storage and indexing
      - Metadata filtering
      - Persistence to disk
      - Querying by nearest neighbours (using Cosine, L2, or IP distance)

Why ChromaDB?
    Unlike raw FAISS, ChromaDB behaves like a database. It stores the actual
    text documents and metadata along with the vectors, makes persistence
    simple, and handles indices and retrieval out of the box.
"""

import os
import json
import numpy as np
import chromadb


class VectorStore:
    """
    Simple ChromaDB-backed vector store that stores:
      - A collection of embeddings, documents (texts), and metadata.

    Usage:
        store = VectorStore(dim=384)
        store.add(embeddings, chunks)       # add numpy array + text list
        results = store.search(query_vec, k=5)  # returns list of (score, text, metadata)
        store.save("my_index")
        store = VectorStore.load("my_index")
    """

    def __init__(
        self,
        dim: int,
        client: chromadb.ClientAPI | None = None, # pyright: ignore[reportPrivateImportUsage]
        collection_name: str = "shakespeare"
    ):
        """
        Args:
            dim: Embedding dimension (must match your Embedder's output).
            client: Optional pre-existing ChromaDB client. If None, creates an EphemeralClient.
            collection_name: Name of the collection in ChromaDB.
        """
        self.dim = dim
        self.collection_name = collection_name
        
        if client is not None:
            self.client = client
        else:
            self.client = chromadb.EphemeralClient()
            
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "dim": self.dim},
        )

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        return self.collection.count()

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

        start_size = self.size
        ids = [f"chunk_{start_size + i}" for i in range(len(texts))]
        
        # ChromaDB expects list of floats (or list of list of floats)
        embeds_list = embeddings.tolist()
        
        # Ensure metadata is valid for ChromaDB (no nested structures, only primitive types)
        formatted_metadata = []
        if metadata:
            for m in metadata:
                formatted_metadata.append(
                    {k: v for k, v in m.items() if isinstance(v, (str, int, float, bool))}
                )
        else:
            formatted_metadata = None

        self.collection.add(
            ids=ids,
            embeddings=embeds_list,
            documents=texts,
            metadatas=formatted_metadata,
        )
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
            score = cosine similarity (range -1 to 1).
        """
        if self.size == 0:
            return []

        if query.ndim == 1:
            query_embeddings = query.tolist()
        else:
            query_embeddings = query[0].tolist()

        # ChromaDB returns sorted list of results (by distance ascending)
        results = self.collection.query(
            query_embeddings=[query_embeddings],
            n_results=k,
        )

        if not results or not results["documents"] or len(results["documents"][0]) == 0:
            return []

        documents = results["documents"][0]
        distances = results["distances"][0] if results["distances"] else [0.0] * len(documents)
        metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)

        search_results = []
        for dist, doc, meta in zip(distances, documents, metadatas):
            # ChromaDB cosine space distance is 1.0 - cosine_similarity
            # Therefore similarity score = 1.0 - distance
            score = 1.0 - dist
            search_results.append((float(score), doc, meta))

        return search_results

    def save(self, directory: str) -> None:
        """
        Persist the index and chunk texts to disk by copying the ephemeral
        collection data into a persistent client.
        """
        os.makedirs(directory, exist_ok=True)

        persistent_client = chromadb.PersistentClient(path=directory)
        
        # Reset the collection at destination directory to start fresh
        try:
            persistent_client.delete_collection(self.collection_name)
        except Exception:
            pass

        persistent_collection = persistent_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "dim": self.dim},
        )

        # Retrieve all items from current (potentially ephemeral) collection
        data = self.collection.get(include=["embeddings", "documents", "metadatas"])
        
        if data["ids"]:
            batch_size = 500
            for i in range(0, len(data["ids"]), batch_size):
                persistent_collection.add(
                    ids=data["ids"][i : i + batch_size],
                    embeddings=data["embeddings"][i : i + batch_size], # pyright: ignore[reportOptionalSubscript]
                    documents=data["documents"][i : i + batch_size], # pyright: ignore[reportOptionalSubscript]
                    metadatas=data["metadatas"][i : i + batch_size], # pyright: ignore[reportOptionalSubscript]
                )

        # Save a meta.json file for backwards compatibility/diagnostics
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump({"dim": self.dim, "size": self.size}, f)

        print(f"VectorStore saved -> {directory}/ ({self.size} vectors, dim={self.dim})")

    @classmethod
    def load(cls, directory: str, collection_name: str = "shakespeare") -> "VectorStore":
        """Load a previously saved VectorStore from a persistent directory."""
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        persistent_client = chromadb.PersistentClient(path=directory)
        
        # Read dimension from meta.json if it exists, otherwise from collection metadata
        dim = 384
        meta_path = os.path.join(directory, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                    dim = meta.get("dim", 384)
            except Exception:
                pass
        else:
            try:
                coll = persistent_client.get_collection(collection_name)
                dim = coll.metadata.get("dim", 384)
            except Exception:
                pass

        store = cls(dim=dim, client=persistent_client, collection_name=collection_name)
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
    import shutil
    tmpdir = tempfile.mkdtemp()
    try:
        store.save(tmpdir)
        # Stop client system to release SQLite database locks on Windows
        if hasattr(store, 'client') and hasattr(store.client, '_system'):
            store.client._system.stop()
            
        store2 = VectorStore.load(tmpdir)
        results2 = store2.search(vecs[0], k=3)
        
        # Stop client system on loaded store too
        if hasattr(store2, 'client') and hasattr(store2.client, '_system'):
            store2.client._system.stop()
            
        assert len(results) == len(results2)
        assert results[0][1] == results2[0][1]
        print("\nSave/load round-trip: OK")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

