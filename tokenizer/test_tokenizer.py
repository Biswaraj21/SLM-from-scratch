from simple_tokenizer import SimpleTokenizer

with open("data/tinystories.txt",'r',encoding='utf-8') as f:
    text=f.read()
    
tokeniser=SimpleTokenizer(text)
print("Vocab size:",tokeniser.vocab_size)

encoded=tokeniser.encoder(text)
#print("Encoded:",encoded)

decoded=tokeniser.decode(encoded)
#print("Decoded:",decoded)