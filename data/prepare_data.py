with open("data/tinystories.txt","r",encoding="utf-8") as f:
    text=f.read()
    
print("Dataset Length:", len(text))
print(text[:500])