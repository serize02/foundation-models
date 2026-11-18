import numpy as np
import torch


class TSDataset(torch.utils.data.Dataset):
    def __init__(self, X, transforms=None):
        self.X = X
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        if self.transforms is not None:
            x = self.transforms(x)
        return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)