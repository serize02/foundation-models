from io import BytesIO

import numpy as np
import pywt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image


class TakeLast:
    def __init__(self, length):
        self.length = length

    def __call__(self, x):
        return x[-self.length:]


class WaveletScalogram:
    def __init__(self, fs=256, wavelet="cmor1.5-1.0", min_frequency=1, max_frequency=65, num_frequencies=100):
        self.fs = fs
        self.wavelet = wavelet
        frequencies = np.geomspace(min_frequency, max_frequency, num_frequencies)
        self.widths = pywt.frequency2scale(wavelet, frequencies / fs)

    def __call__(self, x):
        x = np.asarray(x).squeeze()
        if x.ndim != 1:
            raise ValueError("WaveletScalogram expects a single-channel time series")
        coefficients, frequencies = pywt.cwt(x, self.widths, self.wavelet, sampling_period=1 / self.fs)
        time = np.arange(len(x)) / self.fs
        figure = Figure(figsize=(4, 4))
        FigureCanvasAgg(figure)
        ax = figure.subplots()
        ax.pcolormesh(time, frequencies, np.abs(coefficients), shading="auto", cmap="viridis")
        ax.set_yscale("log", base=2)
        ax.axis("off")
        figure.tight_layout(pad=0)
        buffer = BytesIO()
        figure.savefig(buffer, format="jpg", dpi=224, bbox_inches="tight", pad_inches=0)
        buffer.seek(0)
        with Image.open(buffer) as image:
            return np.asarray(image.convert("RGB")).copy()
