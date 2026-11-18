import numpy as np
import torch
from chronos import ChronosPipeline
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fmlab.datatools.tsdataset import TSDataset


class Chronos:
    def __init__(self, weights, batch_size, transforms=None):
        self.pipeline = ChronosPipeline.from_pretrained(weights)
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "mps" if torch.backends.mps.is_available()
                                   else "cpu")
        self.pipeline.model.to(self.device).eval()
        self.batch_size = batch_size
        self.transforms = transforms

    def embed(self, X):
        loader = DataLoader(dataset=TSDataset(X, self.transforms),
                            batch_size=self.batch_size,
                            shuffle=False)
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(loader, unit="batch"):  # keep inputs on CPU for the tokenizer.
                inputs = batch.permute(0, 2, 1).reshape(-1, batch.shape[1])
                # WARNING: truncates each series to its last context_length observations (512 for chronos-t5-base)
                embedded, _ = self.pipeline.embed(inputs)
                pooled = embedded.mean(dim=1)
                embeddings.append(pooled.reshape(len(batch), -1).float().cpu().numpy())
        return np.concatenate(embeddings)
