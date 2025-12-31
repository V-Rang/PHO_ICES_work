from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class DatasetCreate(Dataset):

    def __init__(self, train_data):
        self.data = np.array(train_data)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]