# FMLab

Foundation models learn general representations through pretraining on large datasets and adapt them to downstream tasks. The approach developed in natural language processing (NLP), with models such as LLaMA, BERT, and GPT, and is now being applied to time series for classification, anomaly detection, forescasting and trend analysis. However time series data hold information that can be represented across different modalities such as scalogram images, which open the possibility of computer vision models to be used as well. This research is centered on studying the potential of LeJEPA in comparison with generalist and domain specific time series foundation models when dealing with physiological signals, specifically in downstreams tasks related to cognitive state estimation. 

> The hypothesis is that LeJEPA's performance highly depends on the downstream task's potential of being represented through the continuous wavelet transform. The CWT exposes a big amount of settings that can explore and we are aiming to do such exploration, although LeJEPA was not trained on first place with scalograms images.

## Introduction

Despite the increasing use of wearable tracking devices, such as smartwatches, wristbands, chest-worn systems that make signal collection easier nowadays, cognitive activity related datasets are hard to label. They usually rely on questionaries, self reports, exposure to environmental conditions, perspective, or subjective criteria that is not generalizable to all subjects within the study. Also, transfer between participants remains difficult because physiological responses vary between individuals and they do not manifest the same at the cognitive level.  However, the corpora of publicly available datasets cover a long range of areas such as stress detection, engagement, concentration levels, sleep stages, academic performance, etc. These conditions motivate the use of representation learning methodologies.

<!-- TODO; find a connection -->

Foundation models with different architectures have been studied for physiological downstreams tasks through the use of clinical and wearable data. Several public available benchmarks recognize that generalist time series foundation models have a comparable performance with domain specific ones under certain settings and downstreams evaluations. Physiological signals are not an exception of this. Although, researchers have created models for specific type of signals only, like PPG, ECG or EDA, some of these models generalize poorly across datasets or are sensitive to changes in sampling frequency and input window length. Their performance can also degrade when applied to signals collected with different devices, which may vary in sensor placement, signal quality, and preprocessing. 

[LeJEPA](https://arxiv.org/abs/2511.08544) is a self-supervised learning approach based on joint embedding predictive architectures. It learns representations by encouraging agreement between different views of the same input and uses Sketched Isotropic Gaussian Regularization (SIGReg) to prevent the representations from collapsing to a constant. This approach has shown competitive performance on image benchmarks, while the related [LeVJEPA](https://arxiv.org/abs/2608.27395) extends the objective to video. In this research, we use an [image-pretrained LeJEPA vision transformer](https://huggingface.co/OK-AI/lejepa-vits16-pretrain-in1k) and explore its application to physiological time series through the continuous wavelet transform.

The CWT provides a bridge between the time series and visual domains by representing the signal as a scalogram, where changes over time and frequency can be observed in an image. The choice of wavelet, frequency range, and window length determines which signal dynamics are presented to the visual encoder. This allows us to explore different representations without retraining the pretrained LeJEPA model, even though it was not originally trained on scalogram images.

## Experiments


**Dataset**. We use the [WESAD dataset](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html), a multimodal dataset collected from 15 participants during a laboratory study on stress and affect detection. The dataset includes physiological and motion signals from chest-worn and wrist-worn devices. For this research, we use the electrocardiogram (ECG) recorded by the chest-worn device and focus on the baseline and stress conditions as a binary classification task.

The chest ECG was originally sampled at 700 Hz and is downsampled to 256 Hz. For each participant and condition, the signal is divided into non-overlapping windows of 2 seconds. These windows are used as the common dataset for all models, with labels identifying the baseline and stress conditions.

**Embedding Models**. We tried these models:

- **[Chronos-T5 base](https://arxiv.org/abs/2403.07815):** Scales and quantizes input values into tokens processed by a Transformer. We average the encoder outputs across the sequence to obtain one embedding per window.

- **[MOMENT-1-large](https://proceedings.mlr.press/v235/goswami24a.html):** Divides the input into patches and processes them with a Transformer encoder. We use it in embedding mode.

- **[NormWear](https://arxiv.org/abs/2412.09758):** Applies the CWT with a Ricker wavelet to the original signal and its first and second differences. The resulting time-frequency representations are divided into patches and processed by a Vision Transformer encoder. Its encoder blocks use self-attention and feed-forward networks, with fusion blocks sharing information between sensor channels. The final classification token provides the signal embedding.

- **[TS-JEPA](https://arxiv.org/abs/2509.25449):** Applies a joint embedding predictive architecture to time series, learning representations by predicting masked parts of a signal from its visible context. We use the last 320 samples of each window as 10 patches of 32 samples, and sum the encoder outputs across patches to obtain the embedding.

- **[LeJEPA ViT-S/16](https://huggingface.co/OK-AI/lejepa-vits16-pretrain-in1k) (complex Morlet, `cmor1.5-1.0`):** Each window is converted into a scalogram using the CWT with 100 frequencies between 1 and 65 Hz. The resulting image is passed to the encoder to obtain an embedding.

**Evaluation Protocols**. We use leave-k--out (k = 3) cross validation through the participants. a logistic regression classifier is trained on embeddings from the remaining participants and evaluated separately for each held-out participant.

- **Linear probing:** The classifier is trained using all available training embeddings. We evaluate accuracy, balanced accuracy, macro F1, Matthews correlation coefficient, and AUROC for each participant.

- **Data scarcity:** The classifier is trained using 1%, 5%, 10%, 25%, 50%, or 100% of the training samples from each class. Sampling is repeated with three seeds, and balanced accuracy and AUROC are averaged across seeds for each participant.

- **Noise stress:** The classifier is trained on clean embeddings and evaluated on clean signals and signals with added Gaussian noise at 0, 10, and 20 dB signal-to-noise ratios. Noisy signals are embedded again before evaluation. Three noise realizations are used, and balanced accuracy and AUROC are averaged across seeds for each participant.

### Results

Results are reported as mean ± sample standard deviation across the 15 participant groups. For data scarcity and noise stress, each participant's score is first averaged across the three seeds; the tables report AUROC.

**Linear probing**. Chronos achieves the highest mean AUROC and balanced accuracy. LeJEPA with the complex Morlet wavelet has an AUROC close to NormWear's, while its balanced accuracy is higher.

| Model | AUROC | Balanced accuracy | Accuracy | Macro F1 | MCC |
|---|---:|---:|---:|---:|---:|
| Chronos-T5 base | 0.883 ± 0.134 | 0.779 ± 0.165 | 0.780 ± 0.178 | 0.749 ± 0.198 | 0.585 ± 0.296 |
| MOMENT-1-large | 0.839 ± 0.192 | 0.744 ± 0.187 | 0.752 ± 0.194 | 0.713 ± 0.217 | 0.515 ± 0.358 |
| LeJEPA ViT-S/16 (complex Morlet) | 0.826 ± 0.154 | 0.729 ± 0.148 | 0.734 ± 0.155 | 0.705 ± 0.170 | 0.474 ± 0.279 |
| NormWear | 0.820 ± 0.187 | 0.678 ± 0.152 | 0.669 ± 0.166 | 0.627 ± 0.189 | 0.383 ± 0.294 |
| TS-JEPA | 0.749 ± 0.243 | 0.670 ± 0.154 | 0.692 ± 0.140 | 0.629 ± 0.183 | 0.356 ± 0.311 |

**Data scarcity**. Chronos has the highest mean AUROC at each tested fraction, reaching 0.842 with 1% of the training samples per class and 0.882 with all training data. LeJEPA with the complex Morlet wavelet reaches 0.803 at 1% and 0.817 at 5%. TS-JEPA shows the largest increase between 1% and 100%, from 0.576 to 0.749. Performance does not increase monotonically for every model.

| Model | 1% | 5% | 10% | 25% | 50% | 100% |
|---|---:|---:|---:|---:|---:|---:|
| Chronos-T5 base | 0.842 ± 0.116 | 0.853 ± 0.123 | 0.847 ± 0.145 | 0.863 ± 0.134 | 0.872 ± 0.132 | 0.882 ± 0.135 |
| MOMENT-1-large | 0.791 ± 0.241 | 0.795 ± 0.240 | 0.806 ± 0.239 | 0.825 ± 0.216 | 0.828 ± 0.206 | 0.839 ± 0.192 |
| LeJEPA ViT-S/16 (complex Morlet) | 0.803 ± 0.150 | 0.817 ± 0.158 | 0.804 ± 0.184 | 0.801 ± 0.185 | 0.811 ± 0.173 | 0.826 ± 0.154 |
| NormWear | 0.749 ± 0.165 | 0.799 ± 0.165 | 0.806 ± 0.173 | 0.814 ± 0.180 | 0.822 ± 0.182 | 0.820 ± 0.187 |
| TS-JEPA | 0.576 ± 0.092 | 0.675 ± 0.192 | 0.703 ± 0.211 | 0.732 ± 0.228 | 0.742 ± 0.236 | 0.749 ± 0.243 |

**Noise stress**. At 0 dB, NormWear has the highest mean AUROC (0.743), followed by TS-JEPA (0.619), LeJEPA with the complex Morlet wavelet (0.610), MOMENT (0.579), and Chronos (0.532). Chronos falls from 0.883 on clean data to 0.532 at 0 dB, while NormWear falls from 0.820 to 0.743.

| Model | Clean | 20 dB | 10 dB | 0 dB |
|---|---:|---:|---:|---:|
| Chronos-T5 base | 0.883 ± 0.134 | 0.789 ± 0.176 | 0.636 ± 0.127 | 0.532 ± 0.085 |
| MOMENT-1-large | 0.839 ± 0.192 | 0.835 ± 0.195 | 0.801 ± 0.193 | 0.579 ± 0.150 |
| LeJEPA ViT-S/16 (complex Morlet) | 0.826 ± 0.154 | 0.787 ± 0.171 | 0.720 ± 0.198 | 0.610 ± 0.121 |
| NormWear | 0.820 ± 0.187 | 0.809 ± 0.193 | 0.778 ± 0.195 | 0.743 ± 0.166 |
| TS-JEPA | 0.749 ± 0.243 | 0.733 ± 0.234 | 0.675 ± 0.189 | 0.619 ± 0.132 |

Scores vary substantially between participants, and these results are descriptive rather than evidence of statistical significance. They are limited to the selected dataset, preprocessing, and model checkpoints.
