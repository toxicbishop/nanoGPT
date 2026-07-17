import torch
from model import GPT
from data import CharTokenizer
import os

path = "checkpoints_v2/ckpt_03000.pt"
ckpt = torch.load(path, map_location="cpu", weights_only=False)
cfg = ckpt["config"]
vocab_size = ckpt["vocab_size"]

model = GPT(cfg, vocab_size=vocab_size)
model.load_state_dict(ckpt["model_state"])
model.eval()

tok_path = "checkpoints_v2/tokenizer.pkl"
tokenizer = CharTokenizer.load(tok_path)

prompt_text = "O Romeo, Romeo!"
ids = torch.tensor(tokenizer.encode(prompt_text), dtype=torch.long).unsqueeze(0)
with torch.no_grad():
    output_ids = model.generate(ids, max_new_tokens=50, temperature=0.8, top_k=40)
generated = tokenizer.decode(output_ids[0])
print(generated)
