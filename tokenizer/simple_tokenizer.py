class SimpleTokenizer:
    def __init__(self,text):
        chars=sorted(list(set(text)))
        self.stoi={ch: i for i,ch in enumerate(chars)}
        self.itos={i: ch for ch,i in self.stoi.items()}
        self.vocab_size=len(chars)
        
    def encoder(self,text):
        return[self.stoi[ch] for ch in text]
    
    def decode(self,tokens):
        return "".join([self.itos[i] for i in tokens])