"""rag/ package init — exposes the main public API."""
from .chunker import chunk_text, chunk_by_paragraph, chunk_file
from .embedder import Embedder
from .store import VectorStore
from .retriever import Retriever
from .pipeline import RAGPipeline, build_prompt

__all__ = [
    "chunk_text", "chunk_by_paragraph", "chunk_file",
    "Embedder",
    "VectorStore",
    "Retriever",
    "RAGPipeline", "build_prompt",
]
