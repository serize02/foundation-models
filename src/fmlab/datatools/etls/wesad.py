import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
import scipy


class WESAD_ETL:
    def __init__(self, root):
        self.root = Path(root)

    def run(
        self, signal, fs, window_size, output_dir, stride=None,
    ):
        X, y, groups = self._load(signal, fs, window_size, stride)

        output_path = Path(output_dir)
        data_path = output_path / "data"
        data_path.mkdir(parents=True, exist_ok=True)
        
        filenames = []
        for i, window in enumerate(X):
            filename = f"{i:05d}.npy"
            np.save(data_path / filename, window)
            filenames.append(f"data/{filename}")
        
        pd.DataFrame({
            "file": filenames, 
            "label": y, 
            "group": groups
        }).to_csv(output_path / "annotations.csv", index=False)

    def _load(
        self, signal, fs, window_size, stride=None
    ):
        X, y, groups = [], [], []
        
        dirs = sorted(path for path in self.root.iterdir() if path.is_dir())
        
        for group_id, obj in enumerate(dirs, start=1):
            base, tsst = self._extract(obj / f"{obj.name}.pkl", signal)
            for label, data in ((0, base), (1, tsst)):
                windows = list(self._transform(data, fs, window_size, stride))
                X.extend(windows)
                y.extend([label] * len(windows))
                groups.extend([group_id] * len(windows))
        
        return X, y, groups

    @staticmethod
    def _extract(pkl_path, signal):
        with pkl_path.open("rb") as file:
            data = pkl.load(file, encoding="latin1")
        
        sig = data["signal"]["chest"][signal]
        labels = data["label"]
        
        base = sig[np.flatnonzero(labels == 1)]
        tsst = sig[np.flatnonzero(labels == 2)]
        
        return base, tsst

    @staticmethod
    def _transform(
        arr, fs, window_size, stride=None
    ):
        arr = scipy.signal.resample_poly(arr, fs, 700)

        stride = stride or window_size
        ws = [arr[i: i + window_size] for i in range(0, len(arr) - window_size + 1, stride)]
        
        return ws
