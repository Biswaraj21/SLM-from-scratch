import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super().__init__()
        
        assert embed_dim%num_heads==0
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.head_dim=embed_dim//num_heads
        
        self.query=nn.Linear(embed_dim,embed_dim)
        self.key=nn.Linear(embed_dim,embed_dim)
        self.value=nn.Linear(embed_dim,embed_dim)
        self.out=nn.Linear(embed_dim,embed_dim)
        
    def forward(self,x):
        batch_size,seq_len,embed_dim=x.shape
        
        Q=self.query(x)
        K=self.key(x)
        V=self.value(x)
        Q=Q.view(batch_size,seq_len,self.num_heads,self.head_dim)
        K=K.view(batch_size,seq_len,self.num_heads,self.head_dim)
        V=V.view(batch_size,seq_len,self.num_heads,self.head_dim)
        
        Q=Q.transpose(1,2)
        K=K.transpose(1,2)
        V=V.transpose(1,2)
        scores=Q@K.transpose(-2,-1)
        scores=scores/(self.head_dim**0.5)
        weights=F.softmax(scores,dim=-1)
        
        out=weights@V
        out=out.transpose(1,2).contiguous()
        out=out.view(batch_size,seq_len,embed_dim)
        out=self.out(out)
        return out