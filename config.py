"""
config.py — Hyperparameters for the modernised nanoGPT.

New in v2:
  - n_kv_heads : GQA (Grouped-Query Attention) — fewer KV heads than Q heads
  - rope_theta : RoPE frequency base (10000 = standard, higher = longer range)
  - norm_eps   : RMSNorm epsilon
  - No positional embedding — RoPE handles position directly in attention
"""

from dataclasses import dataclass, field


@dataclass
class GPTConfig:
    # ── Data ────────────────────────────────────────────────
    data_path: str = "input.txt"
    train_split: float = 0.9

    # ── Model architecture ───────────────────────────────────
    block_size: int = 256          # max context length
    n_embd: int = 256              # embedding / hidden dimension
    n_head: int = 8                # number of QUERY heads
    n_kv_head: int = 2             # number of KEY/VALUE heads  ← GQA knob
    #                                n_kv_head == n_head  → standard MHA
    #                                n_kv_head == 1       → MQA (extreme sharing)
    n_layer: int = 8               # number of transformer blocks (deeper!)
    dropout: float = 0.1
    rope_theta: float = 10000.0   # RoPE base frequency
    norm_eps: float = 1e-6        # RMSNorm epsilon

    # ── Training ─────────────────────────────────────────────
    batch_size: int = 32
    max_iters: int = 3000
    eval_interval: int = 300
    eval_iters: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # ── Logging / checkpointing ──────────────────────────────
    log_interval: int = 100
    checkpoint_dir: str = "checkpoints_v2"
    checkpoint_interval: int = 1000

    # ── Generation ───────────────────────────────────────────
    generate_every: int = 300
    generate_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 40

    # ── System ──────────────────────────────────────────────
    device: str = "auto"
    seed: int = 42

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        )
        assert self.n_head % self.n_kv_head == 0, (
            f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})"
        )
        head_dim = self.n_embd // self.n_head
        assert head_dim % 2 == 0, (
            f"head_dim ({head_dim}) must be even for RoPE"
        )


# ── Presets ──────────────────────────────────────────────────────────────────

# CPU-friendly — ~3.5M params, trains in ~20 min on CPU
# 8 layers deep, GQA 4:1 ratio, RoPE, RMSNorm
TINY_CONFIG = GPTConfig(
    block_size=256,
    n_embd=256,
    n_head=8,
    n_kv_head=2,      # 4 Q heads share each KV head
    n_layer=8,
    dropout=0.1,
    batch_size=16,     # smaller = faster per step on CPU
    max_iters=3000,
    eval_interval=300,
    eval_iters=20,     # fewer batches = faster eval, prints sooner
    generate_every=300,
    log_interval=50,   # print loss more often so terminal feels alive
    checkpoint_dir="checkpoints_v2",
)

# Slightly bigger — ~12M params, needs GPU or patience
SMALL_CONFIG = GPTConfig(
    block_size=512,
    n_embd=512,
    n_head=8,
    n_kv_head=2,
    n_layer=12,
    dropout=0.1,
    batch_size=32,
    max_iters=5000,
    checkpoint_dir="checkpoints_v2_small",
)
