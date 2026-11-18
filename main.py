import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate, to_absolute_path


#----------------------------------------------------------------------------
def run_etl(cfg):
    loader = instantiate(cfg.loader, root=to_absolute_path(cfg.root))
    loader.run(output_dir=to_absolute_path(cfg.output_dir), **cfg.parameters)

#----------------------------------------------------------------------------
def run_embed(cfg):
    dataset_path = Path(to_absolute_path(cfg.dataset.path))
    annotations = pd.read_csv(dataset_path / "annotations.csv")
    X = np.stack([np.load(dataset_path / filename, allow_pickle=False) for filename in annotations["file"]])
    model = instantiate(cfg.model.instance)
    embeddings = model.embed(X)
    output_path = Path(to_absolute_path(cfg.output_dir))
    output_path.mkdir(parents=True, exist_ok=True)
    for filename, embedding in zip(annotations["file"], embeddings):
        embedding_path = output_path / filename
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embedding_path, embedding)
    annotations.to_csv(output_path / "annotations.csv", index=False)

#----------------------------------------------------------------------------
def run_linear_probing(cfg):
    embeddings, annotations = load_embeddings(cfg)
    test = instantiate(cfg.test.instance)
    results = test.run(embeddings, annotations["label"].to_numpy(),
                       annotations["group"].to_numpy())
    save_results(cfg, "results.json", results)

#----------------------------------------------------------------------------
def run_data_scarcity(cfg):
    embeddings, annotations = load_embeddings(cfg)
    test = instantiate(cfg.test.instance)
    results = test.run(embeddings, annotations["label"].to_numpy(), 
                       annotations["group"].to_numpy())
    for label_fraction in cfg.test.instance.label_fractions:
        fraction_results = [result for result in results if result["label_fraction"] == label_fraction]
        save_results(cfg, f"lf_{label_fraction}.json", fraction_results)

#----------------------------------------------------------------------------
def run_noise_stress(cfg):
    embeddings, annotations = load_embeddings(cfg)
    dataset_path = Path(to_absolute_path(cfg.dataset.path))
    X = np.stack([np.load(dataset_path / filename, allow_pickle=False)
                  for filename in annotations["file"]])
    test = instantiate(cfg.test.instance)
    model = instantiate(cfg.model.instance)
    results = test.run(embeddings, annotations["label"].to_numpy(), 
                       annotations["group"].to_numpy(), X, model)
    for snr_db in (None, *cfg.test.instance.snr_levels_db):
        filename = "clean.json" if snr_db is None else f"snr_{snr_db}.json"
        snr_results = [result for result in results if result["snr_db"] == snr_db]
        save_results(cfg, filename, snr_results)

#----------------------------------------------------------------------------
def load_embeddings(cfg):
    embeddings_path = Path(to_absolute_path(cfg.embeddings.path))
    annotations = pd.read_csv(embeddings_path / "annotations.csv")
    embeddings = np.stack([np.load(embeddings_path / filename, allow_pickle=False)
                            for filename in annotations["file"]])
    return embeddings, annotations

def save_results(cfg, filename, results):
    output_path = Path(to_absolute_path(cfg.output_dir))
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / filename).open("w") as output_file:
        json.dump(results, output_file, indent=2)
#----------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="conf")
def main(cfg):
    if cfg.task == "etl":
        run_etl(cfg)
    elif cfg.task == "embed":
        run_embed(cfg)
    elif cfg.task == "eval_class":
        if cfg.test.name == "linear_probing":
            run_linear_probing(cfg)
        elif cfg.test.name == "data_scarcity":
            run_data_scarcity(cfg)
        elif cfg.test.name == "noise_stress":
            run_noise_stress(cfg)


if __name__ == "__main__":
    main()
