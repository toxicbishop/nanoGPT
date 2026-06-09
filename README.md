# nanoGPT — Character-Level Transformer from Scratch

A minimal, heavily-commented GPT implementation in pure PyTorch.
No Hugging Face, no magic — every line is readable and educational.

```
~250 lines of model code  |  ~200 lines of training code
```

---

## Project Structure

```
nanogpt/
├── config.py       ← All hyperparameters (edit this to experiment)
├── data.py         ← Character tokenizer + data loader
├── model.py        ← The GPT model (attention, blocks, generation)
├── train.py        ← Training loop
├── generate.py     ← Load a checkpoint and sample text
└── requirements.txt
```

---

## Quickstart

### 1. Create the virtual environment
```Power Shell
python -m venv .venv
```

### 2. Change execution policy to allow script activation (Windows/PowerShell only)
```Power Shell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Activate the virtual environment
```Power Shell
.venv/Scripts/activate
```

### 3. Install dependencies

```Power Shell
pip install -r requirements.txt
```

### 4. Download a sample corpus (Shakespeare ~1MB)

```Power Shell
python data.py --download shakespeare
# → downloads input.txt
```

Or use **your own text file** — just point `--data` at it.

### 5. Train

```Power Shell
# CPU-friendly tiny model (~3 min on CPU)
python train.py --preset tiny

# Default model (~10M params, needs GPU or ~30 min CPU)
python train.py

# Your own corpus
python train.py --data my_notes.txt --preset tiny
```

Training prints loss every 100 steps and generates text samples every 500 steps:

```
step   500 | train loss 1.8234 | val loss 1.9102 | lr 3.00e-04 | elapsed 42s

──── sample @ step 500 ────
GLOUCESTER:
What, art thou so brave? then let us hear
The king himself hath spoke it; and for my part...
```

### 6. Generate text

```Power Shell
# Single prompt
python generate.py \
    --checkpoint checkpoints/ckpt_02000.pt \
    --prompt "To be or"

# Interactive REPL
python generate.py \
    --checkpoint checkpoints/ckpt_02000.pt \
    --interactive
```

---

## Configuration

Edit `config.py` or pass flags. Key knobs:

| Parameter | Default | Effect |
|---|---|---|
| `block_size` | 256 | Context window. Longer = more memory |
| `n_embd` | 384 | Embedding size. Larger = smarter, slower |
| `n_head` | 6 | Attention heads. Must divide `n_embd` |
| `n_layer` | 6 | Transformer depth |
| `dropout` | 0.2 | Regularisation. 0 for tiny datasets |
| `batch_size` | 64 | Reduce if OOM |
| `learning_rate` | 3e-4 | AdamW LR |
| `temperature` | 0.8 | Generation: higher = more random |
| `top_k` | 40 | 0 = pure sampling, 40 = focused |

### Presets

```python
# In config.py
TINY_CONFIG   # n_embd=128, n_layer=4  — trains in minutes on CPU
SMALL_CONFIG  # n_embd=384, n_layer=6  — ~10M params, needs GPU
```

---

## Architecture Explained

```
Input token IDs  →  Token Embedding
                 +  Positional Embedding
                        ↓
              [TransformerBlock × n_layer]
              ┌─────────────────────────────┐
              │  LayerNorm                  │
              │  MultiHeadSelfAttention ←── │── causal mask prevents
              │  + residual                 │   seeing future tokens
              │  LayerNorm                  │
              │  FeedForward (MLP)       ←──│── per-token processing
              │  + residual                 │
              └─────────────────────────────┘
                        ↓
              LayerNorm  →  Linear head  →  logits (vocab_size)
                        ↓
              softmax  →  P(next character)
```

**Key concepts implemented:**
- **Causal (masked) self-attention** — tokens can only attend to past positions
- **Multi-head attention** — multiple attention patterns in parallel
- **Positional embeddings** — learned position encodings
- **Residual connections** — skip connections around each sub-layer
- **Pre-LayerNorm** — normalise *before* sub-layers (more stable than original)
- **Weight tying** — token embedding and LM head share weights
- **Cosine LR schedule** — warmup + cosine decay
- **Gradient clipping** — prevents exploding gradients
- **Top-k sampling** — controlled randomness at generation time

---

## Learning Path

After you've trained this:

1. **Swap embeddings** — try sinusoidal positional encodings instead of learned
2. **Flash Attention** — `torch.nn.functional.scaled_dot_product_attention` (one line swap, much faster)
3. **BPE tokenization** — replace `CharTokenizer` with `tiktoken` for subword tokens
4. **Fine-tuning** — load a pretrained `gpt2` via Hugging Face, freeze some layers
5. **RAG** — feed retrieved document chunks into the prompt instead of baking knowledge into weights

---

## Resume Training

```Power Shell
python train.py --resume checkpoints/ckpt_02000.pt
```

---

## Expected Loss (Shakespeare)

| Steps | Train Loss | Val Loss | Quality |
|---|---|---|---|
| 0 | ~4.2 | ~4.2 | Random characters |
| 500 | ~2.0 | ~2.1 | Recognisable words |
| 2000 | ~1.5 | ~1.6 | Shakespearean structure |
| 5000 | ~1.2 | ~1.4 | Convincing prose |
