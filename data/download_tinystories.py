from datasets import load_dataset

dataset=load_dataset("roneneldan/TinyStories")

with open("data/tinystories.txt",'w',encoding='utf-8') as f:
    for example in dataset["train"]:
        f.write(example['text']+'\n')