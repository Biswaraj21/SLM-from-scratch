import torch
import torch.nn as nn
from model.multi_head_attention import MultiHeadAttention
from model.feed_forward import FeedForward

class TransfromerBlock(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super().__init__()
        self.attention=MultiHeadAttention(embed_dim,num_heads)
        self.ffn=FeedForward(embed_dim)
        self.ln1=nn.LayerNorm(embed_dim)
        self.ln2=nn.LayerNorm(embed_dim)
        
    def forward(self,x):
        x=x+self.attention(self.ln1(x))
        x=x+self.ffn(self.ln2(x))
        return x
    