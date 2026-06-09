"""
generate.py — Load a trained checkpoint and sample text interactively.

Usage:
    # Generate from the best checkpoint with a custom prompt
    python generate.py --checkpoint checkpoints/ckpt_02000.pt --prompt "To be or"

    # Interactive mode — type a prompt and press Enter
    python generate.py --checkpoint checkpoints/ckpt_02000.pt --interactive

    # Longer, more focused output
    python generate.py --checkpoint checkpoints/ckpt_02000.pt \\
        --tokens 500 --temperature 0.6 --top_k 20
"""

import argparse
import torch
from model import GPT
from data import CharTokenizer
from config import GPTConfig


def load_model(checkpoint_path: str, device: torch.device) -> tuple[GPT, CharTokenizer, GPTConfig]:
    """Load model, tokenizer config from a saved checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    cfg: GPTConfig = ckpt["config"]
    vocab_size: int = ckpt["vocab_size"]

    model = GPT(cfg, vocab_size=vocab_size).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"  Step       : {ckpt['step']}")
    print(f"  Val loss   : {ckpt['val_loss']:.4f}")
    print(f"  Vocab size : {vocab_size}")
    return model, cfg


def generate(
    model: GPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> str:
    # Encode the prompt
    ids = torch.tensor(
        tokenizer.encode(prompt), dtype=torch.long, device=device
    ).unsqueeze(0)   # (1, T)

    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    return tokenizer.decode(out[0])


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from a trained GPT")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--tokenizer", default=None, help="Path to tokenizer.pkl (if different corpus)")
    parser.add_argument("--prompt", default="\n", help="Seed text for generation")
    parser.add_argument("--tokens", type=int, default=300, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k filtering (0 = off)")
    parser.add_argument("--interactive", action="store_true", help="Interactive prompt mode")
    parser.add_argument("--device", default="auto", help="Device: auto / cpu / cuda / mps")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    print(f"Device: {device}\n")

    # Load model
    model, cfg = load_model(args.checkpoint, device)

    # Load tokenizer — must match the corpus the model was trained on
    # Default: look next to the checkpoint file
    import os
    ckpt_dir = os.path.dirname(args.checkpoint)
    tok_path = args.tokenizer or os.path.join(ckpt_dir, "tokenizer.pkl")
    if not os.path.exists(tok_path):
        print(
            f"\n[WARNING] Tokenizer file '{tok_path}' not found.\n"
            "   The tokenizer is embedded in the checkpoint for convenience.\n"
            "   Re-train and save tokenizer with: tokenizer.save('tokenizer.pkl')\n"
            "   Or pass --tokenizer <path>.\n"
        )
        raise SystemExit(1)

    tokenizer = CharTokenizer.load(tok_path)

    # ── Generate ──────────────────────────────────────────────────────────────

    if args.interactive:
        print("\nInteractive mode. Type a prompt and press Enter. Ctrl+C to quit.\n")
        while True:
            try:
                prompt = input(">>> ")
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break
            text = generate(
                model, tokenizer, prompt,
                args.tokens, args.temperature, args.top_k, device
            )
            print(f"\n{text}\n{'─'*60}\n")
    else:
        print(f"\nPrompt: {repr(args.prompt)}\n{'─'*60}")
        text = generate(
            model, tokenizer, args.prompt,
            args.tokens, args.temperature, args.top_k, device
        )
        print(text)
