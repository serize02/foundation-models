import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fmlab.datatools.tsdataset import TSDataset

from .dnn.backbone import NormWear as Backbone
from .dnn.backbone import cwt_wrap


class NormWear:
    def __init__(self, weights, batch_size=64, transforms=None):
        self.batch_size = batch_size
        self.transforms = transforms
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                    else "mps" if torch.backends.mps.is_available()
                                    else "cpu")
        self.model = Backbone()
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.to(self.device).eval()

    def embed(self, X):
        loader = DataLoader(dataset=TSDataset(X, self.transforms),
                            batch_size=self.batch_size, 
                            shuffle=False)
        embeddings = []
        with torch.inference_mode():
            for batch in tqdm(loader, unit="batch"):
                batch = batch.to(self.device)
                inputs = batch.permute(0, 2, 1).reshape(-1, batch.shape[1])
                spec = cwt_wrap(inputs, 0.1, 64) # CWT, first diff, second diff
                spec = spec.reshape(len(batch), batch.shape[2], *spec.shape[1:])
                encoded = self.model.get_signal_embedding(spec.float(), device=self.device)
                embeddings.append(encoded[:, :, 0].reshape(len(batch), -1).cpu().numpy())
        return np.concatenate(embeddings)
