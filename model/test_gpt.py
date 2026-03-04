import torch

from tokenizer.simple_tokenizer import SimpleTokenizer
from model.gpt_model import SLM_model

with open("data/tinystories.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = SimpleTokenizer(text)

sample = "hello world"
tokens = tokenizer.encoder(sample)

x = torch.tensor(tokens).unsqueeze(0)

model = SLM_model(
    vocab_size=tokenizer.vocab_size,
    context_length=32,
    embed_dim=64,
    num_heads=4,
    num_layers=2
)

logits = model(x)

print("Input shape :", x.shape)
print("Output shape:", logits.shape)