"""
train.py — Training loop for the character-level GPT.

Usage:
    # Download sample corpus first
    python data.py --download shakespeare

    # Train with default config
    python train.py

    # Train tiny model (CPU-friendly)
    python train.py --preset tiny

    # Train on your own text file
    python train.py --data my_notes.txt

    # Resume from a checkpoint
    python train.py --resume checkpoints/ckpt_2000.pt
"""

import os
import time
import argparse
import torch

from config import GPTConfig, TINY_CONFIG, SMALL_CONFIG
from data import load_data, get_batch
from model import GPT


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_device(cfg: GPTConfig) -> torch.device:
    if cfg.device == "auto":
        if torch.cuda.is_available():
            dev = "cuda"
        elif torch.backends.mps.is_available():     # Apple Silicon
            dev = "mps"
        else:
            dev = "cpu"
    else:
        dev = cfg.device
    print(f"Using device: {dev}")
    return torch.device(dev)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    cfg: GPTConfig,
) -> dict[str, float]:
    """Average loss over multiple batches — smoother than a single batch."""
    model.eval()
    losses = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        total = 0.0
        for _ in range(cfg.eval_iters):
            x, y = get_batch(data, cfg.block_size, cfg.batch_size)
            _, loss = model(x, y)
            total += loss.item()
        losses[split] = total / cfg.eval_iters
    model.train()
    return losses


def save_checkpoint(model: GPT, optimizer, step: int, val_loss: float, cfg: GPTConfig):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"ckpt_{step:05d}.pt")
    torch.save(
        {
            "step": step,
            "val_loss": val_loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg,
            "vocab_size": model.vocab_size,
        },
        path,
    )
    print(f"  [OK] Checkpoint saved -> {path}")
    return path


def load_checkpoint(path: str, device: torch.device):
    print(f"Resuming from checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: GPTConfig, resume_path: str | None = None):
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_data, val_data, tokenizer = load_data(
        cfg.data_path, cfg.train_split, device
    )

    # Save tokenizer alongside model checkpoints
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    tok_path = os.path.join(cfg.checkpoint_dir, "tokenizer.pkl")
    tokenizer.save(tok_path)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GPT(cfg, vocab_size=tokenizer.vocab_size).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Separate weight decay parameters:
    #   - Apply weight decay to matrices (weights)
    #   - Skip biases and LayerNorm params (scale/shift)
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
    )

    start_step = 0

    # ── Resume ────────────────────────────────────────────────────────────────
    if resume_path:
        ckpt = load_checkpoint(resume_path, device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt["step"] + 1
        print(f"  Resuming from step {start_step}")

    # ── Cosine learning rate schedule ─────────────────────────────────────────
    # Warms up for 100 steps then decays to 10% of peak LR.
    def get_lr(step: int) -> float:
        import math
        warmup = 100
        if step < warmup:
            return cfg.learning_rate * step / warmup
        decay_ratio = (step - warmup) / (cfg.max_iters - warmup)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.learning_rate * 0.1 + coeff * cfg.learning_rate * 0.9

    # ── Prompt for generation samples ─────────────────────────────────────────
    # We'll use a newline as the seed — works for Shakespeare; change as needed
    prompt_text = "\n"
    prompt_ids = torch.tensor(
        tokenizer.encode(prompt_text), dtype=torch.long, device=device
    ).unsqueeze(0)   # (1, T)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n" + "-" * 60)
    print(f"  Training for {cfg.max_iters} steps")
    print(f"  Batch size   : {cfg.batch_size}")
    print(f"  Block size   : {cfg.block_size}")
    print("-" * 60 + "\n")

    best_val_loss = float("inf")
    t0 = time.time()

    for step in range(start_step, cfg.max_iters + 1):

        # ── Adjust LR ────────────────────────────────────────────────────────
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Evaluation ───────────────────────────────────────────────────────
        if step % cfg.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, cfg)
            elapsed = time.time() - t0
            print(
                f"step {step:5d}/{cfg.max_iters} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | "
                f"lr {lr:.2e} | "
                f"elapsed {elapsed:.0f}s"
            )

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(model, optimizer, step, losses["val"], cfg)

        # ── Generate a text sample ────────────────────────────────────────────
        if step % cfg.generate_every == 0 and step > 0:
            model.eval()
            with torch.no_grad():
                sample = model.generate(
                    prompt_ids.clone(),
                    max_new_tokens=cfg.generate_tokens,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                )
            generated = tokenizer.decode(sample[0])
            print(f"\n" + "-" * 40 + f"  sample @ step {step}  " + "-" * 40)
            print(generated)
            print("-" * 90 + "\n")
            model.train()

        # ── Forward + backward ────────────────────────────────────────────────
        x, y = get_batch(train_data, cfg.block_size, cfg.batch_size)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

        # ── Logging ───────────────────────────────────────────────────────────
        if step % cfg.log_interval == 0 and step % cfg.eval_interval != 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e}")

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if step > 0 and step % cfg.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, step, loss.item(), cfg)

    total_time = time.time() - t0
    print(f"\n[DONE] Training complete in {total_time / 60:.1f} min")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints saved in: {cfg.checkpoint_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a character-level GPT")
    parser.add_argument(
        "--preset",
        choices=["tiny", "small"],
        default=None,
        help="Use a preset config (tiny = CPU-friendly, small = ~10M params)",
    )
    parser.add_argument("--data", type=str, default=None, help="Path to text corpus")
    parser.add_argument(
        "--iters", type=int, default=None, help="Override max_iters"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    # Select base config
    if args.preset == "tiny":
        cfg = TINY_CONFIG
    elif args.preset == "small":
        cfg = SMALL_CONFIG
    else:
        cfg = GPTConfig()

    # Apply overrides
    if args.data:
        cfg.data_path = args.data
    if args.iters:
        cfg.max_iters = args.iters

    train(cfg, resume_path=args.resume)
