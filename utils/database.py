import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class MissingPixelDataset(Dataset):
    """
    Creating Dataset for PyTorch model
    """
    def __init__(self, X, y, scaling = False):
        """
        Initialize the input and output target of the network

        Parameters
        ----------
        X : ndarray
            The input data for the neural network
        y : ndarray
            The output target for the neural network
        scaling : bool, optional
            Scaling transformation for the network input, by default False
        """
        if scaling:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        if not torch.is_tensor(X):
            # Change the numpy array to tensor cuda
            self.X = torch.from_numpy(X)
        if not torch.is_tensor(y):
            # Change the numpy array to tensor cuda
            self.y = torch.from_numpy(y)
        # Cast the X, y to tensor float64
        self.X, self.y = self.X.type(torch.float64), self.y.type(torch.float64)

    def __len__(self):
        # calculate the length of the dataset
        return len(self.X)
    
    def __getitem__(self, idx):
        # To get an item inside the dataset
        return self.X[idx], self.y[idx]
    


    