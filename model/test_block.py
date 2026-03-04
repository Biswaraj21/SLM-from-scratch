import torch

from tokenizer.simple_tokenizer import SimpleTokenizer
from model.embedding import TokenEmbedding
from model.positional_embedding import PostionalEmbedding
from model.transformer_block import TransfromerBlock

with open("data/tinystories.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = SimpleTokenizer(text)

sample = "hello world"
tokens = tokenizer.encoder(sample)

x = torch.tensor(tokens).unsqueeze(0)

embed_dim = 32
num_heads = 4
context_length = 32

token_embed = TokenEmbedding(tokenizer.vocab_size, embed_dim)
pos_embed = PostionalEmbedding(context_length,embed_dim)

x = token_embed(x) + pos_embed(x)

block = TransfromerBlock(embed_dim, num_heads)

out = block(x)

print("Input shape :", x.shape)
print("Output shape:", out.shape)