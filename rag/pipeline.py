"""
pipeline.py — Full RAG pipeline: query -> retrieve -> prompt -> LLM -> answer.

This is the complete "Retrieval-Augmented Generation" loop.

Flow:
    user query
        |
    [Retriever]  embed query, search FAISS, return top-k chunks
        |
    [PromptBuilder]  format chunks + query into a prompt string
        |
    [LLM]  generate an answer
        |
    answer string

LLM backend: Ollama (local inference, no API key needed)
    - Install: https://ollama.com
    - Pull a model: `ollama pull llama3.2:1b`  (runs on CPU, ~700MB)
    - Then run this pipeline.

The pipeline degrades gracefully — if Ollama is not running,
it returns just the retrieved context so you can still test retrieval.
"""

import requests # type: ignore
from typing import Optional, Tuple, Union
from .retriever import Retriever
from .security import (
    InputGuardrail,
    OutputGuardrail,
    CanaryTokenManager,
    SecurePromptBuilder,
    GuardrailResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """\
You are a helpful assistant. Use ONLY the context below to answer the question.
If the context does not contain the answer, say "I don't have enough context."

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(
    question: str,
    context_chunks: list[str],
    secure: bool = False,
    canary_token: Optional[str] = None,
) -> str:
    """
    Format chunks and question into prompt.

    Args:
        question:       The user's question.
        context_chunks: List of retrieved text chunks.
        secure:         If True, uses structural XML isolation & delimiter escaping.
        canary_token:   Optional canary token to embed into the system prompt.
    """
    if secure:
        token = canary_token or "canary_default"
        return SecurePromptBuilder.build(question, context_chunks, canary_token=token)

    context = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )
    return PROMPT_TEMPLATE.format(context=context, question=question)


# ─────────────────────────────────────────────────────────────────────────────
# Ollama LLM backend
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"


def call_ollama(
    prompt: str,
    model: str = "llama3.2:1b",
    temperature: float = 0.3,       # lower = more focused for Q&A
    max_tokens: int = 300,
    timeout: int = 60,
) -> str | None:
    """
    Send a prompt to a locally running Ollama instance and return the response.

    Args:
        prompt:      The full prompt string.
        model:       Ollama model tag (e.g. "llama3.2:1b", "mistral", "phi3").
        temperature: Sampling temperature (0 = deterministic, 1 = creative).
        max_tokens:  Max tokens to generate.
        timeout:     HTTP request timeout in seconds.

    Returns:
        Generated text string, or None if Ollama is not available.
    """
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        return None   # Ollama not running — caller handles this
    except Exception as e:
        print(f"  [Ollama error] {e}")
        return None


def is_ollama_running() -> bool:
    """Quick check: is the Ollama server up?"""
    try:
        r = requests.get("http://localhost:11434", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Full RAG Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation.

    Usage:
        rag = RAGPipeline.from_files(["notes.txt", "slides.txt"])
        answer = rag.ask("What is multi-head attention?")
        print(answer)
    """

    def __init__(
        self,
        retriever: Retriever,
        llm_model: str = "llama3.2:1b",
        top_k: int = 4,
        temperature: float = 0.3,
        secure_mode: bool = True,
        input_guardrail: Optional[InputGuardrail] = None,
        output_guardrail: Optional[OutputGuardrail] = None,
    ):
        self.retriever = retriever
        self.llm_model = llm_model
        self.top_k = top_k
        self.temperature = temperature
        self.secure_mode = secure_mode
        self.canary_manager = CanaryTokenManager()
        self.input_guardrail = input_guardrail or InputGuardrail()
        self.output_guardrail = output_guardrail or OutputGuardrail(self.canary_manager)

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_files(
        cls,
        file_paths: list[str],
        index_dir: str | None = None,
        chunk_size: int = 300,
        overlap: int = 60,
        **kwargs,
    ) -> "RAGPipeline":
        """
        Build a RAGPipeline by indexing a list of text files.

        If index_dir is given and the index already exists, loads it from disk
        instead of re-embedding (saves time on repeated runs).
        """
        import os

        # Try to load existing index
        if index_dir and os.path.exists(os.path.join(index_dir, "chroma.sqlite3")):
            print(f"Loading existing index from '{index_dir}' ...")
            retriever = Retriever.load(index_dir, chunk_size=chunk_size, overlap=overlap)
        else:
            # Build fresh index
            retriever = Retriever(chunk_size=chunk_size, overlap=overlap)
            for path in file_paths:
                retriever.add_file(path)
            if index_dir:
                retriever.save(index_dir)

        return cls(retriever=retriever, **kwargs)

    @classmethod
    def from_index(cls, index_dir: str, **kwargs) -> "RAGPipeline":
        """Load a pre-built index from disk."""
        retriever = Retriever.load(index_dir)
        return cls(retriever=retriever, **kwargs)

    # ── Core ask method ───────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        show_sources: bool = False,
        return_metadata: bool = False,
    ) -> Union[str, Tuple[str, dict]]:
        """
        The full secure RAG loop:
            1. [Input Guardrail] Check user query for jailbreaks / injections
            2. [Retriever] Retrieve top-k relevant chunks
            3. [PromptBuilder] Format with XML tags, escape delimiters, inject canary token
            4. [LLM] Generate response
            5. [Output Guardrail] Verify no canary leakage or malicious takeover echo

        Args:
            question:        Natural language question.
            show_sources:    If True, prints retrieved chunks before answering.
            return_metadata: If True, returns (answer, security_telemetry_dict).

        Returns:
            Answer string (or (answer, telemetry) if return_metadata=True).
        """
        security_telemetry = {
            "secure_mode": self.secure_mode,
            "input_guardrail_passed": True,
            "canary_token": None,
            "output_guardrail_passed": True,
            "reasons": [],
        }

        # Step 1: Input Guardrail
        clean_question = question
        if self.secure_mode:
            in_result = self.input_guardrail.evaluate(question)
            if not in_result.passed:
                security_telemetry["input_guardrail_passed"] = False
                security_telemetry["reasons"].append(in_result.reason)
                blocked_msg = f"[SECURITY BLOCK] {in_result.reason}"
                if return_metadata:
                    return blocked_msg, security_telemetry
                return blocked_msg
            clean_question = in_result.sanitized_text

        # Step 2: Retrieve
        results = self.retriever.retrieve(clean_question, k=self.top_k)
        chunks = [r["text"] for r in results]

        if show_sources:
            print(f"\n[Retrieved {len(results)} chunks]")
            for i, r in enumerate(results):
                print(f"  [{i+1}] score={r['score']:.4f} | {r['source']}")
                print(f"       {r['text'][:120].strip()}...")
            print()

        # Step 3: Build Prompt (with Canary Token if secure_mode is active)
        canary = self.canary_manager.generate() if self.secure_mode else None
        security_telemetry["canary_token"] = canary
        prompt = build_prompt(
            clean_question,
            chunks,
            secure=self.secure_mode,
            canary_token=canary,
        )

        # Step 4: Call LLM
        if not is_ollama_running():
            print("[INFO] Ollama not running. Returning retrieved context only.")
            print("       To enable LLM answers: install Ollama, then run:")
            print("         ollama pull llama3.2:1b && ollama serve")
            print()
            context_fallback = "\n\n---\n\n".join(chunks)
            if return_metadata:
                return context_fallback, security_telemetry
            return context_fallback

        raw_answer = call_ollama(prompt, model=self.llm_model, temperature=self.temperature)
        if raw_answer is None:
            err_msg = "[LLM generation failed]"
            if return_metadata:
                return err_msg, security_telemetry
            return err_msg

        # Step 5: Output Guardrail
        final_answer = raw_answer
        if self.secure_mode:
            out_result = self.output_guardrail.evaluate(raw_answer, active_canary=canary)
            if not out_result.passed:
                security_telemetry["output_guardrail_passed"] = False
                security_telemetry["reasons"].append(out_result.reason)
                final_answer = f"[SECURITY INTERCEPTED] {out_result.sanitized_text}"
            else:
                final_answer = out_result.sanitized_text

        if return_metadata:
            return final_answer, security_telemetry
        return final_answer

    def interactive(self):
        """Simple REPL — type questions, get answers."""
        print("\nRAG Pipeline ready. Type a question (or 'quit' to exit).\n")
        while True:
            try:
                q = input("Q: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break
            if q.lower() in ("quit", "exit", "q"):
                break
            if not q:
                continue
            answer = self.ask(q, show_sources=True)
            print(f"\nA: {answer}\n")
            print("-" * 60)


# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    input_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "input.txt")
    index_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rag_index")

    print("=== RAG Pipeline Demo ===\n")

    rag = RAGPipeline.from_files(
        file_paths=[input_file],
        index_dir=index_dir,
        chunk_size=300,
        overlap=60,
    )

    questions = [
        "What does Hamlet say about death?",
        "Who kills the king?",
        "Tell me about love and honour.",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        answer = rag.ask(q, show_sources=False)
        print(f"A: {answer[:300]}...")
        print("-" * 60)
