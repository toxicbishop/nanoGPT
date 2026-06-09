"""
model.py — Modernised character-level GPT (v2)

Upgrades over v1:
  - RMSNorm       replaces LayerNorm   (simpler, no centering, no bias)
  - RoPE          replaces learned positional embeddings (relative, generalises better)
  - GQA           replaces standard MHA (fewer KV heads, less memory, faster)
  - Deeper model  (8 layers vs 4)

This is the architecture family used by LLaMA 2/3, Mistral, Gemma.

Architecture:
    Token Embedding (no position embedding — RoPE handles that inside attention)
        |
    [TransformerBlock x n_layer]
        |  RMSNorm
        |  GroupedQueryAttention  (Q x n_head, K/V x n_kv_head, + RoPE)
        |  + residual
        |  RMSNorm
        |  FeedForward (MLP)
        |  + residual
        |
    RMSNorm
        |
    Linear -> logits (vocab_size)
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig


# ─────────────────────────────────────────────────────────────────────────────
# 1. RMSNorm
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalisation (Zhang & Sennrich, 2019).
    Used in LLaMA, Mistral, Gemma — faster than LayerNorm.

    LayerNorm:  normalise by (x - mean) / std,  then scale + shift
    RMSNorm:    normalise by x / rms(x),         then scale only

    No bias, no mean subtraction = fewer ops, same quality.
    rms(x) = sqrt( mean(x^2) + eps )
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # learnable scale (gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ─────────────────────────────────────────────────────────────────────────────
# 2. Rotary Positional Embeddings (RoPE)
# ─────────────────────────────────────────────────────────────────────────────

def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Pre-compute the complex rotation factors for RoPE.

    Intuition:
      - Standard pos embeddings add a position vector to the token.
      - RoPE instead *rotates* the Q and K vectors based on position.
      - If token A is at position p and B is at position q,
        their attention score naturally encodes (p - q), the *relative* distance.
      - This means the model generalises better to unseen sequence lengths.

    Returns: freqs_cis of shape (max_seq_len, head_dim // 2), complex64
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    # Frequencies for each pair of dimensions: theta^(-2i/d) for i in 0..d/2
    # Low-frequency pairs rotate slowly (encode long-range), high-freq = short-range
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )  # (head_dim // 2,)

    positions = torch.arange(max_seq_len, device=device).float()  # (max_seq_len,)
    freqs_outer = torch.outer(positions, freqs)                    # (T, head_dim // 2)

    # Convert to complex: e^(i * theta) = cos(theta) + i*sin(theta)
    freqs_cis = torch.polar(torch.ones_like(freqs_outer), freqs_outer)  # complex64
    return freqs_cis  # (max_seq_len, head_dim // 2)


def apply_rope(
    xq: torch.Tensor,        # (B, T, n_heads, head_dim)
    xk: torch.Tensor,        # (B, T, n_kv_heads, head_dim)
    freqs_cis: torch.Tensor, # (max_seq_len, head_dim // 2)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to Q and K by rotating pairs of dimensions.

    The rotation for position p on dimension pair (2i, 2i+1) is:
        [cos(p * freq_i)  -sin(p * freq_i)] [q_2i  ]
        [sin(p * freq_i)   cos(p * freq_i)] [q_2i+1]

    This is equivalent to element-wise complex multiplication.
    """

    def rotate(x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_heads, head_dim)
        T = x.shape[1]
        # Reshape last dim into pairs: (..., head_dim) -> (..., head_dim//2, 2)
        x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        # View as complex: (..., head_dim//2) complex
        x_complex = torch.view_as_complex(x_pairs)
        # freqs_cis: (T, head_dim//2) -> (1, T, 1, head_dim//2) for broadcasting
        f = freqs_cis[:T].unsqueeze(0).unsqueeze(2)
        # Rotate: complex multiply
        x_rotated = torch.view_as_real(x_complex * f).flatten(start_dim=3)
        return x_rotated.type_as(x)

    return rotate(xq), rotate(xk)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Grouped-Query Attention (GQA)
# ─────────────────────────────────────────────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (Ainslie et al., 2023) — used in LLaMA 2/3, Mistral.

    Standard MHA:    n_head Q heads,    n_head K heads,    n_head V heads
    GQA:             n_head Q heads,  n_kv_head K heads,  n_kv_head V heads
    MQA (extreme):   n_head Q heads,       1    K head,       1    V head

    Each group of (n_head // n_kv_head) Q heads SHARES a single K/V head.
    This cuts the KV cache size by (n_head / n_kv_head)x — huge at inference.
    Quality is nearly identical to full MHA for the same training budget.

    Example with n_head=8, n_kv_head=2:
        Q heads: [Q0 Q1 Q2 Q3] [Q4 Q5 Q6 Q7]
        K heads:      [K0    ]       [K1    ]
        V heads:      [V0    ]       [V1    ]
        Groups 0-3 share K0/V0, groups 4-7 share K1/V1.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.groups = config.n_head // config.n_kv_head  # Q heads per KV head

        # Separate projections for Q, K, V
        self.wq = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)),
        )

    def forward(
        self,
        x: torch.Tensor,              # (B, T, n_embd)
        freqs_cis: torch.Tensor,      # (max_seq_len, head_dim // 2)
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # ── Project Q, K, V ──────────────────────────────────────────────────
        q = self.wq(x).view(B, T, self.n_head, self.head_dim)      # (B,T,Hq,D)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim)   # (B,T,Hkv,D)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim)   # (B,T,Hkv,D)

        # ── Apply RoPE to Q and K ────────────────────────────────────────────
        # V is NOT rotated — RoPE only goes on the keys/queries that compute
        # the attention scores, not on the values that get aggregated.
        q, k = apply_rope(q, k, freqs_cis)

        # ── Expand K, V to match n_head (GQA repeat) ─────────────────────────
        # (B, T, n_kv_head, head_dim) -> (B, T, n_head, head_dim)
        # Each KV head is repeated `groups` times to pair with its Q group.
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=2)
            v = v.repeat_interleave(self.groups, dim=2)

        # ── Scaled dot-product attention ─────────────────────────────────────
        # Transpose to (B, n_head, T, head_dim) for batch matmul
        q = q.transpose(1, 2)   # (B, Hq,  T, D)
        k = k.transpose(1, 2)   # (B, Hq,  T, D)  (after expand)
        v = v.transpose(1, 2)   # (B, Hq,  T, D)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale   # (B, Hq, T, T)

        # Causal mask
        attn = attn.masked_fill(self.causal_mask[:T, :T] == 0, float("-inf")) # type: ignore
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Aggregate values
        out = attn @ v                                          # (B, Hq, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, n_embd)

        return self.wo(out)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Feed-Forward (same as v1, GELU MLP)
# ─────────────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Two-layer MLP with 4x expansion.

    Future upgrade: swap to SwiGLU (used in LLaMA) for a free quality boost:
        gate = silu(linear1(x))
        up   = linear2(x)
        out  = gate * up  -> linear3
    For now keeping GELU to stay minimal.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = 4 * config.n_embd
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, config.n_embd, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Transformer Block
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block with:
      - RMSNorm (not LayerNorm)
      - GroupedQueryAttention with RoPE (not standard MHA)
      - FeedForward MLP
      - Residual connections around each sub-layer
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.norm2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freqs_cis)   # communicate
        x = x + self.ffn(self.norm2(x))               # compute
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 6. GPT v2 — full model
# ─────────────────────────────────────────────────────────────────────────────

class GPT(nn.Module):
    """
    Modernised character-level GPT with RMSNorm + GQA + RoPE.

    Key difference from v1:
      - No positional embedding table. Position info enters via RoPE
        inside each attention layer, applied directly to Q and K.
      - RoPE freqs are precomputed once at init and reused every forward pass.
    """

    def __init__(self, config: GPTConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        head_dim = config.n_embd // config.n_head

        self.tok_emb = nn.Embedding(vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.norm_f = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)

        # Weight tying — token embedding and LM head share weights
        self.tok_emb.weight = self.lm_head.weight

        # Pre-compute RoPE frequencies once (stored as buffer, moves with .to(device))
        freqs = precompute_rope_freqs(head_dim, config.block_size, config.rope_theta)
        self.register_buffer("freqs_cis", freqs)

        # Weight initialisation
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("net.2.weight"):
                # Scale residual projections down by sqrt(2 * n_layer)
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT v2 initialised - {n_params / 1e6:.2f}M parameters")
        print(f"  Layers    : {config.n_layer}")
        print(f"  d_model   : {config.n_embd}")
        print(f"  Q heads   : {config.n_head}")
        print(f"  KV heads  : {config.n_kv_head}  (GQA ratio {config.n_head // config.n_kv_head}:1)")
        print(f"  head_dim  : {head_dim}")
        print(f"  RoPE theta: {config.rope_theta}")

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        idx: torch.Tensor,                    # (B, T)
        targets: Optional[torch.Tensor] = None, # (B, T)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Sequence {T} > block_size {self.config.block_size}"
        )

        x = self.drop(self.tok_emb(idx))   # (B, T, n_embd) — no pos embedding!

        for block in self.blocks:
            x = block(x, self.freqs_cis)   # RoPE freqs passed through each block

        x = self.norm_f(x)
        logits = self.lm_head(x)           # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """Autoregressive generation — identical API to v1."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)

        return idx
