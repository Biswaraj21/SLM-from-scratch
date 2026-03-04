import torch
import torch.nn as nn

class PostionalEmbedding(nn.Module):
    def __init__(self,context_length,embed_dim):
        super().__init__()
        
        self.postion_embedding=nn.Embedding(context_length,embed_dim)
        
    def forward(self,x):
        batch_size,seq_len=x.shape
        positions=torch.arange(seq_len,device=x.device)
        positions=positions.unsqueeze(0).expand(batch_size,seq_len)
        return self.postion_embedding(positions)