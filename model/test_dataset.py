from tokenizer.simple_tokenizer import SimpleTokenizer
from model.dataset import LanguageDataset

with open('data/tinystories.txt','r',encoding='utf-8') as f:
    text=f.read()
    
tokeniser=SimpleTokenizer(text)
encoded=tokeniser.encoder(text)

dataset=LanguageDataset(encoded,context_length=8)

x,y=dataset[0]
# print('Input:',x)
# print("Text:", tokeniser.decode(x.tolist()))
# print("Target:",y)
# print("Tetx:",tokeniser.decode(y.tolist()))