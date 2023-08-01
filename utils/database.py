import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class MissingPixelDataset(Dataset):
    def __init__(self, X, y, scaling = False):
        if scaling:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)
        self.X, self.y = self.X.type(torch.float64), self.y.type(torch.float64)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    


    