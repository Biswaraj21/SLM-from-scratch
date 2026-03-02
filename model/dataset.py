import torch

class LanguageDataset:
    def __init__(self,data,context_length):
        self.data=torch.tensor(data,dtype=torch.long)
        self.context_length=context_length
        
    def __len__(self):
        return len(self.data)-self.context_length
    
    def __getitem__(self,idx):
        x=self.data[idx:idx+self.context_length]
        y=self.data[idx+1:idx+self.context_length+1]
        
        return x,y