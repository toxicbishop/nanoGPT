"""
retriever.py — Ties Embedder + VectorStore into a single retrieval interface.

This is the "R" in RAG.

Full flow:
    Documents (text files)
         |
    chunker.chunk_file()            <- split into overlapping text windows
         |
    Embedder.embed(chunks)          <- each chunk -> 384-dim vector
         |
    VectorStore.add(vecs, chunks)   <- store in FAISS index
         |
         v  [index built, ready to query]

    User query (string)
         |
    Embedder.embed_one(query)       <- query -> 384-dim vector
         |
    VectorStore.search(vec, k=5)    <- ANN search -> top-k (score, chunk) pairs
         |
    Return top-k text chunks        -> passed to LLM as context
"""

import os
from typing import Optional

from .chunker import chunk_file, chunk_text
from .embedder import Embedder
from .store import VectorStore


class Retriever:
    """
    High-level interface for building and querying a RAG index.

    Typical workflow:
        r = Retriever()
        r.add_file("notes.txt")
        r.add_file("lecture2.txt")
        r.save("my_index")

        # Later:
        r = Retriever.load("my_index")
        chunks = r.retrieve("What is attention mechanism?", k=3)
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 256,
        overlap: int = 64,
    ):
        self.embedder = Embedder(model_name)
        self.store = VectorStore(dim=self.embedder.dim)
        self.chunk_size = chunk_size
        self.overlap = overlap

    # ── Indexing ─────────────────────────────────────────────────────────────

    def add_text(
        self,
        text: str,
        source: str = "inline",
        chunk_mode: str = "fixed",
    ) -> int:
        """
        Chunk, embed, and index a raw string.

        Args:
            text:       The document text.
            source:     A label stored in metadata (e.g. filename).
            chunk_mode: "fixed" or "paragraph".

        Returns:
            Number of chunks added.
        """
        if chunk_mode == "paragraph":
            from .chunker import chunk_by_paragraph
            chunks = chunk_by_paragraph(text, max_chunk_size=self.chunk_size)
        else:
            chunks = chunk_text(text, self.chunk_size, self.overlap)

        if not chunks:
            print(f"  [WARNING] No chunks produced from '{source}'")
            return 0

        print(f"  Embedding {len(chunks)} chunks from '{source}' ...")
        vectors = self.embedder.embed(chunks, show_progress=len(chunks) > 50)
        metadata = [{"source": source, "chunk_id": i} for i in range(len(chunks))]
        self.store.add(vectors, chunks, metadata)
        return len(chunks)

    def add_file(
        self,
        path: str,
        chunk_mode: str = "fixed",
    ) -> int:
        """
        Load a .txt file and add it to the index.

        Args:
            path:       Path to a UTF-8 text file.
            chunk_mode: "fixed" or "paragraph".

        Returns:
            Number of chunks added.
        """
        assert os.path.exists(path), f"File not found: {path}"
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.add_text(text, source=os.path.basename(path), chunk_mode=chunk_mode)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> list[dict]:
        """
        Find the top-k most relevant chunks for a query string.

        Args:
            query: Natural language question or search string.
            k:     Number of results to return.

        Returns:
            List of dicts: [{"score": float, "text": str, "source": str}, ...]
            Sorted by relevance (highest score first).
        """
        query_vec = self.embedder.embed_one(query)
        raw = self.store.search(query_vec, k=k)

        return [
            {
                "score": score,
                "text": text,
                "source": meta.get("source", "?"),
                "chunk_id": meta.get("chunk_id", -1),
            }
            for score, text, meta in raw
        ]

    def retrieve_as_context(self, query: str, k: int = 5, sep: str = "\n\n---\n\n") -> str:
        """
        Retrieve top-k chunks and join them into a single context string
        ready to be stuffed into an LLM prompt.
        """
        results = self.retrieve(query, k=k)
        return sep.join(r["text"] for r in results)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Save the index to disk so you don't have to re-embed next time."""
        self.store.save(directory)

    @classmethod
    def load(
        cls,
        directory: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 256,
        overlap: int = 64,
    ) -> "Retriever":
        """Load a saved index from disk."""
        r = cls.__new__(cls)
        r.embedder = Embedder(model_name)
        r.store = VectorStore.load(directory)
        r.chunk_size = chunk_size
        r.overlap = overlap
        return r


# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    input_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "input.txt")
    if not os.path.exists(input_file):
        print("Run from nanogpt/ and make sure input.txt exists (python data.py --download shakespeare)")
        raise SystemExit(1)

    print("Building index from Shakespeare corpus ...")
    r = Retriever(chunk_size=300, overlap=60)
    r.add_file(input_file)
    print(f"\nIndex built: {r.store.size} chunks\n")

    queries = [
        "What does the king say about loyalty?",
        "A sword fight and blood",
        "Love and marriage",
    ]

    for q in queries:
        print(f"Query: '{q}'")
        results = r.retrieve(q, k=3)
        for i, res in enumerate(results):
            print(f"  [{i+1}] score={res['score']:.4f} | {res['text'][:100].strip()}...")
        print()
