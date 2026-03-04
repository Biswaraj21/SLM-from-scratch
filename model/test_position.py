import torch
from tokenizer.simple_tokenizer import SimpleTokenizer
from model.embedding import TokenEmbedding
from model.positional_embedding import PostionalEmbedding

with open("data/tinystories.txt",'r',encoding='utf-8') as f:
    text=f.read()
    
tokeniser=SimpleTokenizer(text)
sample="hello world"
tokens=tokeniser.encoder(sample)
x=torch.tensor(tokens).unsqueeze(0)

embed_dim=32
context_length=32

token_embed=TokenEmbedding(tokeniser.vocab_size,embed_dim)
pos_embed=PostionalEmbedding(context_length,embed_dim)

token_vectors=token_embed(x)
positon_vectors=pos_embed(x)

final=token_vectors+positon_vectors
print("Token shape:",token_vectors.shape)
print("Postional shape:",positon_vectors.shape)
print("Final:",final.shape)