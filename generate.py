import torch
import torch.nn.functional as F
from tokenizer.simple_tokenizer import SimpleTokenizer
from model.gpt_model import SLM_model

with open("data/tinystories.txt",'r',encoding='utf-8') as f:
    text=f.read()
    
tokeniser=SimpleTokenizer(text)
model=SLM_model(vocab_size=tokeniser.vocab_size,context_length=32,embed_dim=32,num_heads=2,num_layers=1)
model.load_state_dict(torch.load("model.pt"))
model.eval()

def generate(model,tokeniser,prompt,max_new_token=100):
    token=tokeniser.encoder(prompt)
    tokens=torch.tensor(token).unsqueeze(0)
    for _ in range(max_new_token):
        tokens_cond=tokens[:-32]
        logits=model(tokens_cond)
        logits=logits[:,-1,:]
        probs=F.softwax(logits,dim=-1)
        next_token=torch.multinomial(probs,num_samples=1)
        tokens=torch.cat([tokens,next_token.unsqueeze(0)],dim=-1)
    return tokeniser.decoder(tokens.squeeze().tolist())

print(generate(model,tokeniser, "Once upon"))