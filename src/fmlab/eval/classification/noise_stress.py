import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tqdm.auto import tqdm

from .linear_probing import LinearProbingTest


class NoiseStressTest(LinearProbingTest):
    def __init__(self, leave_k_out, scale_embeddings, snr_levels_db, seeds=(1, 2, 3)):
        super().__init__(leave_k_out, scale_embeddings)
        self.snr_levels_db = snr_levels_db
        self.seeds = seeds

    def run(self, X_embeddings, y, groups, X, model):
        X_embeddings = np.asarray(X_embeddings)
        y = np.asarray(y)
        groups = np.asarray(groups)
        X = np.asarray(X)
        generators = [np.random.default_rng(seed) for seed in self.seeds]
        results = []

        for train_idx, test_idx in tqdm(self.leave_k_out(groups)):
            clf = self._get_clf()
            clf.fit(X_embeddings[train_idx], y[train_idx])
            positive_idx = np.flatnonzero(clf.classes_ == 1)[0]
            X_test = X[test_idx].astype(float)
            y_test = y[test_idx]
            test_groups = groups[test_idx]
            X_rms = np.sqrt(np.mean(np.square(X_test), axis=1, keepdims=True))
            noises = []
            for rng in generators:
                noise = rng.standard_normal(X_test.shape)
                noise_rms = np.sqrt(np.mean(np.square(noise), axis=1, keepdims=True))
                noises.append(np.divide(noise * X_rms, noise_rms, out=np.zeros_like(noise), where=noise_rms != 0))

            for snr_db in (None, *self.snr_levels_db):
                balanced_scores = {group: [] for group in np.unique(test_groups)}
                auroc_scores = {group: [] for group in np.unique(test_groups)}
                for noise in ([None] if snr_db is None else noises):
                    features = X_embeddings[test_idx] if noise is None else model.embed(X_test + noise * 10.0 ** (-snr_db / 20.0))
                    y_pred = clf.predict(features)
                    y_proba = clf.predict_proba(features)[:, positive_idx]
                    for group in np.unique(test_groups):
                        mask = test_groups == group
                        balanced_scores[group].append(balanced_accuracy_score(y_test[mask], y_pred[mask]))
                        if len(np.unique(y_test[mask])) == 2:
                            auroc_scores[group].append(roc_auc_score(y_test[mask], y_proba[mask]))
                for group in np.unique(test_groups):
                    results.append({
                        "snr_db": snr_db,
                        "group": str(group),
                        "group_results": {
                            "balanced_accuracy": float(np.mean(balanced_scores[group])),
                            "auroc": float(np.mean(auroc_scores[group])) if auroc_scores[group] else None,
                        },
                    })
        return results
