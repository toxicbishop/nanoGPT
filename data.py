"""
data.py — Character-level tokenizer + dataset utilities.

The simplest possible tokenizer: each unique character in your
corpus gets an integer ID. No BPE, no sentencepiece — just chars.

Vocab size for Shakespeare (~65), Wikipedia slice (~150), code (~100).
"""

import os
import torch
import pickle
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

class CharTokenizer:
    """
    Minimal character-level tokenizer.

    encode("hello") → [46, 43, 50, 50, 53]
    decode([46, 43, 50, 50, 53]) → "hello"
    """

    def __init__(self, text: str):
        self.chars = sorted(set(text))          # all unique characters
        self.vocab_size = len(self.chars)
        self._stoi = {ch: i for i, ch in enumerate(self.chars)}   # char → int
        self._itos = {i: ch for i, ch in enumerate(self.chars)}   # int → char

    def encode(self, text: str) -> list[int]:
        return [self._stoi[c] for c in text]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self._itos[i] for i in ids)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"chars": self.chars}, f)
        print(f"Tokenizer saved -> {path}")

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "rb") as f:
            data = pickle.load(f)
        tok = cls.__new__(cls)
        tok.chars = data["chars"]
        tok.vocab_size = len(tok.chars)
        tok._stoi = {ch: i for i, ch in enumerate(tok.chars)}
        tok._itos = {i: ch for i, ch in enumerate(tok.chars)}
        return tok


# ─────────────────────────────────────────────────────────────────────────────
# Dataset preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_data(
    data_path: str,
    train_split: float = 0.9,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, CharTokenizer]:
    """
    Read raw text, tokenize, and split into train / validation tensors.

    Returns:
        train_data  — 1-D LongTensor of token IDs (training portion)
        val_data    — 1-D LongTensor of token IDs (validation portion)
        tokenizer   — fitted CharTokenizer
    """
    assert os.path.exists(data_path), (
        f"Data file not found: {data_path}\n"
        "Run `python data.py --download shakespeare` to grab a sample corpus."
    )

    print(f"Loading corpus from '{data_path}' ...")
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"  Corpus size : {len(text):,} characters")

    tokenizer = CharTokenizer(text)
    print(f"  Vocab size  : {tokenizer.vocab_size} unique characters")

    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(train_split * len(ids))
    train_data = ids[:n]
    val_data = ids[n:]

    print(f"  Train tokens: {len(train_data):,}")
    print(f"  Val tokens  : {len(val_data):,}")

    return train_data.to(device), val_data.to(device), tokenizer


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a random batch of (input, target) pairs.

    For each sequence in the batch:
        x = tokens[i : i+block_size]        ← input context
        y = tokens[i+1 : i+block_size+1]    ← shifted by 1 = next-token targets

    This is the core of autoregressive language modelling.
    """
    # Random starting positions
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


# ─────────────────────────────────────────────────────────────────────────────
# CLI helper — download a sample corpus
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_CORPORA = {
    "shakespeare": (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "input.txt",
    ),
    "bible": (
        "https://raw.githubusercontent.com/gavin19/kjv-bible/master/kjv.txt",
        "bible.txt",
    ),
}


if __name__ == "__main__":
    import argparse
    import urllib.request

    parser = argparse.ArgumentParser(description="Download a sample text corpus")
    parser.add_argument(
        "--download",
        choices=list(SAMPLE_CORPORA.keys()),
        default="shakespeare",
        help="Which corpus to download",
    )
    args = parser.parse_args()

    url, fname = SAMPLE_CORPORA[args.download]
    print(f"Downloading {args.download} corpus -> {fname} ...")
    urllib.request.urlretrieve(url, fname)
    print(f"Done! ({os.path.getsize(fname):,} bytes)")
    print(f"\nNow run:  python train.py")
