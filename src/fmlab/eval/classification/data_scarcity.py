import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tqdm.auto import tqdm

from .linear_probing import LinearProbingTest


class DataScarcityTest(LinearProbingTest):
    def __init__(self, leave_k_out, scale_embeddings, label_fractions, seeds=(1, 2, 3)):
        super().__init__(leave_k_out, scale_embeddings)
        self.label_fractions = label_fractions
        self.seeds = seeds

    def run(self, X_embeddings, y, groups):
        X_embeddings = np.asarray(X_embeddings)
        y = np.asarray(y)
        groups = np.asarray(groups)
        generators = [np.random.default_rng(seed) for seed in self.seeds]
        results = []
        for train_idx, test_idx in tqdm(self.leave_k_out(groups)):
            shuffled_indices = [rng.permutation(train_idx) for rng in generators]
            y_test = y[test_idx]
            test_groups = groups[test_idx]
            for label_fraction in self.label_fractions:
                balanced_scores = {group: [] for group in np.unique(test_groups)}
                auroc_scores = {group: [] for group in np.unique(test_groups)}
                for shuffled_idx in shuffled_indices:
                    selected = []
                    for label in np.unique(y[shuffled_idx]):
                        indices = shuffled_idx[y[shuffled_idx] == label]
                        n = max(1, int(np.ceil(len(indices) * label_fraction)))
                        selected.extend(indices[:n])
                    selected = np.asarray(selected, dtype=int)
                    clf = self._get_clf()
                    clf.fit(X_embeddings[selected], y[selected])
                    y_pred = clf.predict(X_embeddings[test_idx])
                    positive_idx = np.flatnonzero(clf.classes_ == 1)[0]
                    y_proba = clf.predict_proba(X_embeddings[test_idx])[:, positive_idx]
                    for group in np.unique(test_groups):
                        mask = test_groups == group
                        balanced_scores[group].append(balanced_accuracy_score(y_test[mask], y_pred[mask]))
                        if len(np.unique(y_test[mask])) == 2:
                            auroc_scores[group].append(roc_auc_score(y_test[mask], y_proba[mask]))
                for group in np.unique(test_groups):
                    results.append({
                        "label_fraction": label_fraction,
                        "group": str(group),
                        "group_results": {
                            "balanced_accuracy": float(np.mean(balanced_scores[group])),
                            "auroc": float(np.mean(auroc_scores[group])) if auroc_scores[group] else None,
                        },
                    })
        return results
