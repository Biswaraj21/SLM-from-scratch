import torch
import torch.nn as nn
from model.embedding import TokenEmbedding
from model.positional_embedding import PostionalEmbedding
from model.transformer_block import TransfromerBlock

class SLM_model(nn.Module):
    def __init__(self,vocab_size,context_length,embed_dim,num_heads,num_layers):
        super().__init__()
        self.token_embedding=TokenEmbedding(vocab_size,embed_dim)
        self.pos_embedding=PostionalEmbedding(context_length,embed_dim)
        self.blocks=nn.ModuleList([TransfromerBlock(embed_dim,num_heads) for _ in range(num_layers)])
        self.ln_final=nn.LayerNorm(embed_dim)
        self.head=nn.Linear(embed_dim,vocab_size)
        
    def forward(self,x):
        x=self.token_embedding(x)+self.pos_embedding(x)
        for block in self.blocks:
            x=block(x)
        x=self.ln_final(x)
        logits=self.head(x)
        return logits