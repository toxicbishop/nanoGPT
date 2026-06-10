"""
demo.py — End-to-end RAG demo you can run right now.

What this does, step by step:
    1. Loads the Shakespeare corpus (input.txt)
    2. Chunks it into 300-char overlapping windows
    3. Embeds all chunks with all-MiniLM-L6-v2 (downloads once, ~90MB)
    4. Stores vectors in a FAISS index (saved to rag_index/ for reuse)
    5. Takes your questions, embeds them, retrieves top-k chunks
    6. If Ollama is running: sends to llama3.2:1b for a real answer
       If not: prints the retrieved chunks so you see what RAG retrieved

Run:
    cd nanogpt
    pip install -r rag/requirements.txt
    python rag/demo.py

    # Optional: enable LLM answers
    # 1. Install Ollama from https://ollama.com
    # 2. ollama pull llama3.2:1b
    # 3. ollama serve        (keep running in another terminal)
    # 4. python rag/demo.py  (now gives real answers)
"""

import os
import sys
import time

# Allow running as either:
#   python rag/demo.py        (from nanogpt/)
#   python demo.py            (from nanogpt/rag/)
_here = os.path.dirname(os.path.abspath(__file__))          # .../nanogpt/rag
_root = os.path.dirname(_here)                               # .../nanogpt
sys.path.insert(0, _root)   # so `import rag` resolves from nanogpt/

from rag.retriever import Retriever
from rag.pipeline import RAGPipeline, is_ollama_running

# ── Paths ─────────────────────────────────────────────────────────────────────
CORPUS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "input.txt")
INDEX_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rag_index")
#──────────────────────────────────────────────────────────────────────────────


def print_separator(title: str = "", width: int = 60):
    if title:
        pad = (width - len(title) - 2) // 2
        print("-" * pad + f" {title} " + "-" * pad)
    else:
        print("-" * width)


def run_demo():
    print_separator("RAG Demo", 60)
    print("Corpus :", CORPUS_PATH)
    print("Index  :", INDEX_DIR)
    print()

    # ── Step 1-4: Build or load index ─────────────────────────────────────────
    index_exists = os.path.exists(os.path.join(INDEX_DIR, "chroma.sqlite3"))

    if index_exists:
        print("[1/4] Loading existing ChromaDB index (skip re-embedding) ...")
        rag = RAGPipeline.from_index(INDEX_DIR, top_k=4)
    else:
        print("[1/4] Building ChromaDB index from corpus ...")
        print("      This embeds ~4000 chunks — takes ~2 min on CPU, once only.\n")
        t0 = time.time()
        rag = RAGPipeline.from_files(
            file_paths=[CORPUS_PATH],
            index_dir=INDEX_DIR,
            chunk_size=300,
            overlap=60,
            top_k=4,
        )
        print(f"\nIndex built in {time.time()-t0:.1f}s and saved to '{INDEX_DIR}/'")

    print(f"\nIndex ready: {rag.retriever.store.size} chunks indexed\n")

    # ── Step 5-6: Show retrieval-only results ─────────────────────────────────
    print_separator("Retrieval Demo (no LLM needed)")

    test_queries = [
        "What does Hamlet say about death and dying?",
        "A betrayal and murder of a king",
        "Love and romantic feelings",
    ]

    for q in test_queries:
        print(f"\nQuery: '{q}'")
        results = rag.retriever.retrieve(q, k=3)
        for i, r in enumerate(results):
            score_bar = "#" * int(r["score"] * 20)
            print(f"  [{i+1}] score={r['score']:.4f} {score_bar}")
            print(f"       {r['text'][:150].strip()}...")

    # ── Step 6: LLM-augmented answers ─────────────────────────────────────────
    print()
    print_separator("LLM-Augmented Answers")

    if is_ollama_running():
        print("Ollama detected! Sending queries to llama3.2:1b ...\n")
        llm_questions = [
            "What is the central theme of Hamlet?",
            "How does power lead to corruption in the plays?",
        ]
        for q in llm_questions:
            print(f"Q: {q}")
            answer = rag.ask(q, show_sources=True)
            print(f"A: {answer}\n")
            print_separator()
    else:
        print("Ollama is not running — skipping LLM answers.")
        print()
        print("To get real LLM-powered answers:")
        print("  1. Download Ollama  ->  https://ollama.com/download")
        print("  2. Run:  ollama pull llama3.2:1b")
        print("  3. Run:  ollama serve       (keep open in another terminal)")
        print("  4. Run:  python rag/demo.py (this script, again)")

    # ── Interactive mode ──────────────────────────────────────────────────────
    print()
    print_separator("Interactive Mode")
    print("The index is ready. Ask your own questions!")
    print("(Retrieval works even without Ollama)\n")
    rag.interactive()


if __name__ == "__main__":
    if not os.path.exists(CORPUS_PATH):
        print(f"ERROR: Corpus not found at '{CORPUS_PATH}'")
        print("Run first:  python data.py --download shakespeare")
        sys.exit(1)
    run_demo()
