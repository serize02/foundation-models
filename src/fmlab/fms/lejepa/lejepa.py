import sys

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel

from fmlab.datatools.tsdataset import TSDataset


class LeJEPA:
    def __init__(self, model_id, batch_size=8, transforms=None):
        self.batch_size = batch_size
        self.transforms = transforms
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "mps" if torch.backends.mps.is_available()
                                   else "cpu")
        model_path = snapshot_download(repo_id=model_id)
        if model_path not in sys.path:
            sys.path.insert(0, model_path)
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(self.device).eval()

    def embed(self, X):
        loader = DataLoader(dataset=TSDataset(X, self.transforms),
                            batch_size=self.batch_size, shuffle=False)
        embeddings = []
        with torch.inference_mode():
            for batch in tqdm(loader, unit="batch"):
                images = [image.numpy().astype(np.uint8) for image in batch]
                inputs = self.processor(images=images, return_tensors="pt")
                outputs = self.model(inputs["pixel_values"].to(self.device))
                embeddings.append(outputs["latent"].cpu().numpy())
        return np.concatenate(embeddings)
