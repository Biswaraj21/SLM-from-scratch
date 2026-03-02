import torch
from tokenizer.simple_tokenizer import SimpleTokenizer
from model.embedding import TokenEmbedding

with open("data/tinystories.txt",'r',encoding='utf-8') as f:
    text=f.read()
    
tokeniser=SimpleTokenizer(text)
sample='hello world'
tokens=tokeniser.encoder(sample)

x=torch.tensor(tokens).unsqueeze(0)
embedding=TokenEmbedding(tokeniser.vocab_size,embed_dim=32)

out=embedding(x)

print("Input shape:",x.shape)

print("Output shape:",out.shape)