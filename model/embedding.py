import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self,vocal_size,embed_dim):
        super().__init__()
        
        self.embedding=nn.Embedding(vocal_size,embed_dim)
        
    def forward(self,x):
        return self.embedding(x)