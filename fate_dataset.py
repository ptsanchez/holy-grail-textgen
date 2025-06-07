import torch
from torch.utils.data import Dataset

class FateDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        xlen = self.X.size(dim=0)
        ylen = self.y.size(dim=0)
        if len(self.X) != len(self.y):
            print(f"FateDataset Warning: X:({xlen}) and y:({ylen}) are different lengths")
        return xlen
    
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])