import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.embed_dim=embed_dim
        self.query=nn.Linear(embed_dim,embed_dim)
        self.key=nn.Linear(embed_dim,embed_dim)
        self.value=nn.Linear(embed_dim,embed_dim)
        
    def forward(self,x):
        batch_size,seq_len,embed_dim=x.shape
        Q=self.query(x)
        K=self.key(x)
        V=self.value(x)
        
        scores=Q@K.transpose(-2,-1)
        scores=scores/(self.embed_dim**0.5)
        
        mask=torch.tril(torch.ones(seq_len,seq_len,device=x.device))
        scores=scores.masked_fill(mask==0,float('-inf'))
        weights=F.softmax(scores,dim=-1)
        out=weights@V
        return out