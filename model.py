"""
model.py — The full GPT model, built from scratch.

Architecture (each piece explained inline):

    Input tokens
        ↓
    Token Embedding  +  Positional Embedding
        ↓
    [TransformerBlock] × n_layer
        │  ├─ LayerNorm
        │  ├─ MultiHeadSelfAttention   ← the "look at other tokens" part
        │  ├─ LayerNorm
        │  └─ FeedForward (MLP)        ← the "think per-token" part
        ↓
    LayerNorm
        ↓
    Linear head  →  logits over vocab
        ↓
    softmax → probability distribution over next character
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GPTConfig


# ─────────────────────────────────────────────────────────────────────────────
# 1. Causal Self-Attention (single head)
# ─────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    One head of masked (causal) self-attention.

    Each token attends to all *preceding* tokens (and itself),
    but CANNOT see future tokens — that would be cheating at prediction.

    The mask looks like:
        token 0 → can see [0]
        token 1 → can see [0, 1]
        token 2 → can see [0, 1, 2]
        ...
    """

    def __init__(self, config: GPTConfig, head_size: int):
        super().__init__()
        self.head_size = head_size

        # Q, K, V projections — no bias (common in modern transformers)
        self.q = nn.Linear(config.n_embd, head_size, bias=False)
        self.k = nn.Linear(config.n_embd, head_size, bias=False)
        self.v = nn.Linear(config.n_embd, head_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        # Lower-triangular mask — registered as a buffer (not a parameter)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape   # Batch, Time (seq len), Channels (embd dim)

        q = self.q(x)   # (B, T, head_size)
        k = self.k(x)   # (B, T, head_size)
        v = self.v(x)   # (B, T, head_size)

        # ── Scaled dot-product attention ────────────────────────────────────
        #
        #   Attention(Q, K, V) = softmax( QKᵀ / √d ) · V
        #
        # Dividing by √head_size keeps the dot products from growing too large,
        # which would push softmax into regions with tiny gradients.

        scale = self.head_size ** -0.5
        attn = q @ k.transpose(-2, -1) * scale   # (B, T, T)

        # Apply causal mask: future positions get -inf → softmax gives 0
        attn = attn.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values
        out = attn @ v   # (B, T, head_size)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Multi-Head Attention
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    Run n_head independent attention heads in parallel, then concatenate.

    Why multiple heads? Each head can specialise in a different relationship
    (e.g., one head tracks subject-verb agreement, another tracks
    coreference). The model learns what's useful automatically.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, (
            "n_embd must be divisible by n_head"
        )
        head_size = config.n_embd // config.n_head

        self.heads = nn.ModuleList(
            [CausalSelfAttention(config, head_size) for _ in range(config.n_head)]
        )
        # Project concatenated heads back to n_embd
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each head: (B, T, head_size) → concat → (B, T, n_embd)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feed-Forward Network (MLP)
# ─────────────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    A simple two-layer MLP applied *independently* to each token position.

    Attention aggregates information across positions.
    This MLP then "thinks" about what was gathered at each position.

    The 4× expansion factor is the original Transformer convention.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),                              # GELU ≈ smoother ReLU
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Transformer Block
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    One transformer block = LayerNorm + Attention + LayerNorm + FFN,
    each with a residual (skip) connection.

    Residual connections:
        x = x + attention(norm(x))   ← "communicate"
        x = x + ffn(norm(x))         ← "compute"

    Pre-norm (norm before the sub-layer) trains more stably than
    the original post-norm design from Attention Is All You Need.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # communicate
        x = x + self.ffn(self.ln2(x))    # compute
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 5. GPT — putting it all together
# ─────────────────────────────────────────────────────────────────────────────

class GPT(nn.Module):
    """
    Character-level GPT.

    Given a sequence of character IDs, predicts the probability distribution
    over the next character at every position simultaneously.
    (This is how teacher forcing works during training.)
    """

    def __init__(self, config: GPTConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        self.transformer = nn.ModuleDict(
            {
                "token_emb": nn.Embedding(vocab_size, config.n_embd),
                # Positional embedding: each position 0..block_size-1 gets a vector.
                # The model learns what "being at position 5" means.
                "pos_emb": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "blocks": nn.ModuleList(
                    [TransformerBlock(config) for _ in range(config.n_layer)]
                ),
                "ln_f": nn.LayerNorm(config.n_embd),   # final layer norm
            }
        )

        # Language model head: project from n_embd → vocab_size logits
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)

        # Weight tying: share weights between token embedding and lm_head.
        # This is standard practice — reduces params and improves performance.
        self.transformer["token_emb"].weight = self.lm_head.weight

        # Initialise weights sensibly
        self.apply(self._init_weights)
        # Scale residual projections by 1/√(2 * n_layer) — GPT-2 trick
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT initialised - {n_params / 1e6:.2f}M parameters")

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,                     # (B, T) — token IDs
        targets: torch.Tensor | None = None,   # (B, T) — next-token targets
    ) -> Tuple:
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        # ── Embeddings ───────────────────────────────────────────────────────
        tok_emb = self.transformer["token_emb"](idx)               # (B, T, n_embd)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.transformer["pos_emb"](pos)                 # (T, n_embd)

        x = self.transformer["drop"](tok_emb + pos_emb)            # (B, T, n_embd)

        # ── Transformer blocks ───────────────────────────────────────────────
        for block in self.transformer["blocks"]:
            x = block(x)

        x = self.transformer["ln_f"](x)                            # (B, T, n_embd)
        logits = self.lm_head(x)                                   # (B, T, vocab_size)

        # ── Loss ─────────────────────────────────────────────────────────────
        loss = None
        if targets is not None:
            # Reshape for cross-entropy: (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,          # (B, T) — seed / prompt tokens
        max_new_tokens: int = 200,
        temperature: float = 1.0,   # > 1 → more random, < 1 → more focused
        top_k: int = 0,             # 0 = disabled; otherwise keep top-k logits
    ) -> torch.Tensor:
        """
        Autoregressively sample new tokens one at a time.

        At each step:
            1. Forward pass to get logits for the last position
            2. Optionally apply top-k filtering
            3. Sample from the resulting distribution
            4. Append sampled token and repeat
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size if it grew too long
            idx_cond = idx[:, -self.config.block_size :]

            logits, _ = self(idx_cond)
            # Take logits at the last position (the "next token" slot)
            logits = logits[:, -1, :] / temperature     # (B, vocab_size)

            if top_k > 0:
                # Zero out all logits except the top-k
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_token], dim=1)              # (B, T+1)

        return idx

