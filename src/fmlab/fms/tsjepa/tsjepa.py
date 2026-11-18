import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fmlab.datatools.tsdataset import TSDataset

from .dnn.encoder import Encoder


class TSJEPA:
    def __init__(self, weights, batch_size=64, transforms=None):
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                    else "mps" if torch.backends.mps.is_available()
                                    else "cpu")
        self.batch_size = batch_size
        self.transforms = transforms
        self.model = Encoder(num_patches=10, dim_in=32, kernel_size=3, embed_dim=128, nhead=2, num_layers=1)
        checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
        self.model.load_state_dict(checkpoint['encoder'], strict=True)
        self.model.to(self.device).eval().requires_grad_(False)

    def embed(self, X):
        loader = DataLoader(dataset=TSDataset(X, self.transforms),
                            batch_size=self.batch_size, 
                            shuffle=False)
        embeddings = []
        with torch.inference_mode():
            for batch in tqdm(loader, unit="batch"):
                batch = batch.to(self.device)
                tokens = batch.permute(0, 2, 1).reshape(len(batch) * batch.shape[2], self.model.num_patches, 32) # dim_in
                output = self.model(tokens).sum(dim=1)
                embeddings.append(output.reshape(len(batch), -1).cpu().numpy())
        return np.concatenate(embeddings)
