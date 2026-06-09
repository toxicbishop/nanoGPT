"""
config.py — All hyperparameters in one place.
Tweak these to experiment with model size / training speed.
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    # ── Data ────────────────────────────────────────────────
    data_path: str = "input.txt"       # path to your raw text corpus
    train_split: float = 0.9           # fraction used for training

    # ── Model architecture ───────────────────────────────────
    block_size: int = 256              # context window (max sequence length)
    n_embd: int = 384                  # embedding dimension
    n_head: int = 6                    # number of attention heads (n_embd % n_head == 0)
    n_layer: int = 6                   # number of transformer blocks
    dropout: float = 0.2               # dropout probability

    # ── Training ─────────────────────────────────────────────
    batch_size: int = 64               # sequences per batch
    max_iters: int = 5000              # total training steps
    eval_interval: int = 500           # how often to evaluate on val set
    eval_iters: int = 200              # batches to average for eval loss
    learning_rate: float = 3e-4        # AdamW learning rate
    weight_decay: float = 0.1
    grad_clip: float = 1.0             # clip gradients at this norm

    # ── Logging / checkpointing ──────────────────────────────
    log_interval: int = 100            # print loss every N steps
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1000    # save checkpoint every N steps

    # ── Generation ───────────────────────────────────────────
    generate_every: int = 500          # sample from model every N steps
    generate_tokens: int = 200         # number of tokens to generate
    temperature: float = 0.8           # sampling temperature (higher = more random)
    top_k: int = 40                    # top-k sampling (0 = disabled)

    # ── System ──────────────────────────────────────────────
    device: str = "auto"               # "auto", "cpu", "cuda", "mps"
    seed: int = 42


# Tiny config — trains in minutes on CPU, good for quick experiments
TINY_CONFIG = GPTConfig(
    block_size=128,
    n_embd=128,
    n_head=4,
    n_layer=4,
    dropout=0.1,
    batch_size=32,
    max_iters=3000,
    eval_interval=300,
    generate_every=300,
)

# Small config — ~10M params, needs GPU or ~30 min on CPU
SMALL_CONFIG = GPTConfig(
    block_size=256,
    n_embd=384,
    n_head=6,
    n_layer=6,
    dropout=0.2,
    batch_size=64,
    max_iters=5000,
)
