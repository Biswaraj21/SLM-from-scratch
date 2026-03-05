from collections import Counter

class BPETokenizer:
    def __init__(self,vocab_size=500):
        self.vocal_size=vocab_size
        self.merges={}
        self.stoi={}
        self.itos={}
        
    def get_stats(self,words):
        pairs=Counter()
        for word in words:
            symbols=word.split()
            for i in range(len(symbols)-1):
                pairs[(symbols[i],symbols[i+1])]+=1
        return pairs
    
    def merge_vocab(self,pair,words):
        bigram=" ".join(pair)
        replacement="".join(pair)
        new_words=[]
        for word in words:
            new_word=word.replace(bigram,replacement)
            new_words.append(new_word)
        return new_words
    
    def train(self,text):
        words=text.split()
        words=[" ".join(list(word))+" </w" for word in words]
        vocab=set()
        for word in words:
            vocab.update(word.split())
        while len(vocab)<self.vocal_size:
            pairs=self.get_stats(words)
            if not pairs:
                break
            best=max(pairs,key=pairs.get)
            words=self.merge_vocab(best,words)
            self.merges[best]="".join(best)
            vocab=set()
            for word in words:
                vocab.update(word.split())
        vocab=sorted(vocab)
        self.stoi={tok:i for i,tok in enumerate(vocab)}
        self.itos={i:tok for tok,i in self.stoi.items()}
    
    def encode(self,text):
        tokens=[]
        for word in text.split():
            symbols=list(word)+["</w>"]
            i+=1
            while i<len(symbols)-1:
                pair=(symbols[i],symbols[i+1])
                if pair in self.merges:
                    symbols[i:i+2]=["".join(pair)]
                else:
                    i+=1
            tokens.extend(symbols)
        return [self.stoi[t] for t in tokens if t in self.stoi]
    
    def decode(self,ids):
        tokens=[self.itos[i] for i in ids]
        text=""
        for tok in tokens:
            if tok=="</w>":
                text+=" "
            else:
                text+=tok
        return text.strip()