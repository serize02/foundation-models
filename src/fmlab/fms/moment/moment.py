import numpy as np
import torch
from momentfm import MOMENTPipeline
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fmlab.datatools.tsdataset import TSDataset


class MOMENT:
    def __init__(self, weights, batch_size=64, transforms=None):
        self.batch_size = batch_size
        self.transforms = transforms
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                    else "mps" if torch.backends.mps.is_available()
                                    else "cpu")
        self.model = MOMENTPipeline.from_pretrained(
            weights,
            model_kwargs={"task_name": "embedding"}
        )
        self.model.init()
        self.model.to(self.device).eval()

    def embed(self, X):
        loader = DataLoader(dataset=TSDataset(X, self.transforms),
                            batch_size=self.batch_size,
                            shuffle=False)
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(loader, unit="batch"):
                batch = batch.to(self.device).permute(0, 2, 1)
                embedded = self.model(x_enc=batch).embeddings
                embeddings.append(embedded.cpu().numpy().reshape(len(batch), -1))
        return np.concatenate(embeddings)
