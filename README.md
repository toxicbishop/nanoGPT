# nanoGPT — Character-Level Transformer from Scratch

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![RAG](https://img.shields.io/badge/RAG-FAISS%20%2B%20SentenceTransformers-green.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)

A CPU-friendly character-level GPT from scratch in PyTorch featuring modern architecture (RoPE, RMSNorm, GQA) and a modular RAG pipeline with FAISS.
No Hugging Face, no magic — every line is readable and educational.

```
~250 lines of model code  |  ~200 lines of training code
```

---

## System Architecture

![nano GPT System Architecture](assets/nano%20GPT-System%20Architecture.png)

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

---

## Stage 2: Retrieval-Augmented Generation (RAG)

A fully modular RAG pipeline that allows you to perform semantic search and context injection over your corpus using `FAISS` and `sentence-transformers`.

### 1. Install RAG dependencies
```Power Shell
pip install -r rag/requirements.txt
```

### 2. Run the Interactive RAG Demo
```Power Shell
python rag/demo.py
```

### Sample Demo Output
```
------------------------- RAG Demo -------------------------
Corpus : D:\Code\Repo\SLM\nanogpt\input.txt
Index  : D:\Code\Repo\SLM\nanogpt\rag_index

[1/4] Building FAISS index from corpus ...
      This embeds ~4000 chunks — takes ~2 min on CPU, once only.

Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
  Embedding dim : 384
  Embedding 4648 chunks from 'input.txt' ...
  Added 4648 chunks | Total: 4648
VectorStore saved -> D:\Code\Repo\SLM\nanogpt\rag_index/ (4648 vectors, dim=384)

Index built in 51.6s and saved to 'D:\Code\Repo\SLM\nanogpt\rag_index/'

Index ready: 4648 chunks indexed

-------------- Retrieval Demo (no LLM needed) --------------

Query: 'What does Romeo say about Juliet and the sun?'
  [1] score=0.6661 #############
       me, shall we go?

BENVOLIO:
Go, then; for 'tis in vain
To seek him here that means not to be found.

ROMEO:
He jests at scars that never felt a wound.
But, soft! what light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already

Query: 'Tell me about Juliet's poison and death.'
  [1] score=0.6608 #############
       As I intended, for it wrought on her
The form of death: meantime I writ to Romeo,
That he should hither come as this dire night,
To help to take her from her borrow'd grave,
Being the time the potion's force should cease.
But he which bore my letter, Friar John,
Was stay'd by accident, and yesternig
```

To run with full LLM-augmented generation:
1. Download [Ollama](https://ollama.com)
2. Run `ollama pull llama3.2:1b`
3. Start the Ollama server: `ollama serve`
4. Run `python rag/demo.py` again.

