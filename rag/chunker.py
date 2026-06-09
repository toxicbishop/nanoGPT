"""
chunker.py — Split raw text into overlapping chunks for embedding.

Why chunk?
    Embedding models have a token limit (~256-512 tokens).
    A full document won't fit. We split it into smaller pieces
    so each chunk can be embedded into a single dense vector.

Why overlap?
    If a sentence spans a chunk boundary, neither chunk captures it fully.
    Overlapping means the boundary content appears in both neighbours,
    so retrieval doesn't miss it.

    chunk 0: [===========|------]
    chunk 1:          [---|============]
                      ^ overlap region
"""


def chunk_text(
    text: str,
    chunk_size: int = 256,      # max characters per chunk
    overlap: int = 64,           # characters shared between adjacent chunks
    min_chunk_size: int = 50,    # discard chunks shorter than this (e.g. trailing whitespace)
) -> list[str]:
    """
    Split text into overlapping fixed-size character chunks.

    Args:
        text:           Raw input string.
        chunk_size:     Target size of each chunk in characters.
        overlap:        How many characters to repeat between adjacent chunks.
        min_chunk_size: Chunks shorter than this are dropped.

    Returns:
        List of chunk strings.

    Example:
        text = "ABCDEFGHIJ"
        chunk_size=4, overlap=1
        -> ["ABCD", "DEFG", "GHIJ"]
    """
    assert chunk_size > overlap, "chunk_size must be greater than overlap"
    step = chunk_size - overlap
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)
        start += step
    return chunks


def chunk_by_paragraph(
    text: str,
    max_chunk_size: int = 512,
    overlap_paragraphs: int = 1,
) -> list[str]:
    """
    Split text on blank lines (paragraphs), then merge small paragraphs
    so each chunk is at most max_chunk_size characters.

    Better than fixed-size chunking for structured documents (essays, chapters)
    because it respects natural boundaries.

    overlap_paragraphs: how many paragraphs from the previous chunk to
                        prepend to the next one for context continuity.
    """
    # Split on one or more blank lines
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current = []
    current_len = 0

    for i, para in enumerate(paragraphs):
        if current_len + len(para) > max_chunk_size and current:
            chunks.append("\n\n".join(current))
            # Keep the last N paragraphs as overlap for next chunk
            current = current[-overlap_paragraphs:]
            current_len = sum(len(p) for p in current)

        current.append(para)
        current_len += len(para)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def chunk_file(
    path: str,
    mode: str = "fixed",
    **kwargs,
) -> list[str]:
    """
    Load a text file and chunk it.

    Args:
        path:   Path to .txt file.
        mode:   "fixed" (character windows) or "paragraph" (blank-line splits).
        kwargs: Passed to chunk_text() or chunk_by_paragraph().

    Returns:
        List of chunk strings.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if mode == "paragraph":
        return chunk_by_paragraph(text, **kwargs)
    return chunk_text(text, **kwargs)


# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles.

Thus conscience does make cowards of us all;
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought.
    """.strip()

    chunks = chunk_text(sample, chunk_size=100, overlap=20)
    print(f"Fixed-size chunking: {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        print(f"\n  [{i}] ({len(c)} chars): {c[:60]}...")

    print()
    chunks_p = chunk_by_paragraph(sample, max_chunk_size=200)
    print(f"Paragraph chunking: {len(chunks_p)} chunks")
    for i, c in enumerate(chunks_p):
        print(f"\n  [{i}]: {c[:80]}...")
