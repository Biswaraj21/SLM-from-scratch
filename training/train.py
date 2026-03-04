import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tokenizer.simple_tokenizer import SimpleTokenizer
from model.dataset import LanguageDataset
from model.gpt_model import SLM_model

with open("data/tinystories.txt",'r',encoding='utf-8') as f:
    text=f.read()
    
tokeniser=SimpleTokenizer(text)
encoded=tokeniser.encoder(text[:1_000_000])
context_length=32
dataset=LanguageDataset(encoded,context_length)
loader=DataLoader(dataset,batch_size=4,shuffle=True)

model=SLM_model(vocab_size=tokeniser.vocab_size,context_length=context_length,embed_dim=32,num_heads=2,num_layers=1)
optimizer=optim.AdamW(model.parameters(),lr=3e-4)
criterion=nn.CrossEntropyLoss()

epochs=3
for epoch in range(epochs):
    total_loss=0
    for x,y in loader:
        logits=model(x)
        B,T,C=logits.shape
        logits=logits.view(B*T,C)
        y=y.view(B*T)
        loss=criterion(logits,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
    
torch.save(model.state_dict(),"model.pt")
print("Model saved")