import streamlit as st
import torch
import os

from model import GPT
from data import CharTokenizer
from config import GPTConfig
try:
    from rag.retriever import Retriever
except ImportError:
    Retriever = None

st.set_page_config(page_title="nanoGPT + RAG", layout="wide")
st.title("nanoGPT & Semantic Search")

# --- Globals ---
CKPT_DIR = "checkpoints_v2"
INDEX_DIR = "rag_index"

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not os.path.exists(CKPT_DIR):
        return None, None, None
    files = sorted([f for f in os.listdir(CKPT_DIR) if f.endswith(".pt")])
    if not files:
        return None, None, None
    path = os.path.join(CKPT_DIR, files[-1])
    
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    vocab_size = ckpt["vocab_size"]
    
    model = GPT(cfg, vocab_size=vocab_size)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    tok_path = os.path.join(CKPT_DIR, "tokenizer.pkl")
    if os.path.exists(tok_path):
        tokenizer = CharTokenizer.load(tok_path)
    else:
        tokenizer = None
        
    return model, cfg, tokenizer

@st.cache_resource(show_spinner="Loading RAG index...")
def load_retriever():
    if Retriever is None:
        return None
    if not os.path.exists(INDEX_DIR) or not os.path.exists(os.path.join(INDEX_DIR, "chroma.sqlite3")):
        return None
    return Retriever.load(INDEX_DIR)

model, cfg, tokenizer = load_model()
retriever = load_retriever()

if model is None or tokenizer is None:
    st.error(f"No checkpoint or tokenizer found in `{CKPT_DIR}/`. Train the model first.")
    st.stop()

# Show model stats
st.sidebar.markdown("### Model Stats")
st.sidebar.markdown(f"- **Vocab size**: {model.vocab_size}")
st.sidebar.markdown(f"- **Layers**: {cfg.n_layer}")
st.sidebar.markdown(f"- **Embedding**: {cfg.n_embd}")
st.sidebar.markdown(f"- **Heads**: {cfg.n_head} (KV: {cfg.n_kv_head})")

tab1, tab2 = st.tabs(["Text Generation", "Semantic RAG Search"])

with tab1:
    st.markdown("### Generate text from scratch")
    prompt = st.text_input("Prompt", value="To be or not")
    col1, col2, col3 = st.columns(3)
    max_tokens = col1.slider("Tokens to generate", 100, 1000, 300)
    temperature = col2.slider("Temperature", 0.1, 2.0, 0.8)
    top_k = col3.slider("Top-k", 0, 100, 40)

    if st.button("Generate", key="gen_button"):
        with st.spinner("Generating..."):
            ids = torch.tensor(
                tokenizer.encode(prompt), dtype=torch.long
            ).unsqueeze(0)

            with torch.no_grad():
                output_ids = model.generate(
                    ids, max_new_tokens=max_tokens,
                    temperature=temperature, top_k=top_k
                )

            generated = tokenizer.decode(output_ids[0])
            st.text_area("Output", value=generated, height=300)

with tab2:
    st.markdown("### Semantic Search & RAG")
    if retriever is None:
        st.warning(f"No RAG index found in `{INDEX_DIR}/` or dependencies missing. Please run `rag/demo.py` locally first.")
    else:
        query = st.text_input("Question / Search Query", value="What does Hamlet say about death?")
        
        col1, col2 = st.columns(2)
        top_k_chunks = col1.slider("Number of chunks to retrieve", 1, 10, 3)
        use_nanogpt = col2.checkbox("Pass context to nanoGPT", value=True, help="Injects chunks into the prompt. Note: nanoGPT isn't instruction-tuned, so it will continue the text in its own style.")
        
        if st.button("Search", key="rag_button"):
            with st.spinner("Searching ChromaDB..."):
                results = retriever.retrieve(query, k=top_k_chunks)
                
            st.markdown("#### Retrieved Context")
            for i, r in enumerate(results):
                with st.expander(f"Chunk {i+1} (Score: {r['score']:.4f})"):
                    st.write(r['text'])
            
            if use_nanogpt:
                with st.spinner("nanoGPT is trying to answer..."):
                    context_text = "\n\n".join(r["text"] for r in results)
                    rag_prompt = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
                    
                    ids = torch.tensor(
                        tokenizer.encode(rag_prompt), dtype=torch.long
                    ).unsqueeze(0)
                    
                    with torch.no_grad():
                        output_ids = model.generate(
                            ids, max_new_tokens=200,
                            temperature=0.7, top_k=40
                        )
                    
                    full_text = tokenizer.decode(output_ids[0])
                    # slice out the prompt for display
                    answer_only = full_text[len(rag_prompt):]
                    st.markdown("#### nanoGPT Answer")
                    st.text_area("Generated Answer", value=answer_only, height=200)
