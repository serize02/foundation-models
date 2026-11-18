
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


class LinearProbingTest:
    def __init__(self, leave_k_out, scale_embeddings):
        self.leave_k_out = leave_k_out
        self.scale_embeddings = scale_embeddings

    def run(self, X_embeddings, y, groups):
        X_embeddings = np.asarray(X_embeddings)
        y = np.asarray(y)
        groups = np.asarray(groups)
        results = []
        for train_idx, test_idx in tqdm(self.leave_k_out(groups)):
            X_train, X_test = X_embeddings[train_idx], X_embeddings[test_idx]
            y_train, y_test, test_groups = y[train_idx], y[test_idx], groups[test_idx]
            clf = self._get_clf()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            positive_idx = np.flatnonzero(clf.classes_ == 1)[0]
            y_proba = clf.predict_proba(X_test)[:, positive_idx]
            for group in np.unique(test_groups):
                mask = test_groups == group
                results.append({
                    "group": str(group),
                    "group_results": self._evaluate(y_test[mask], y_pred[mask], y_proba[mask], clf.classes_)
                })
        return results

    @staticmethod
    def _evaluate(y_true, y_pred, y_proba, classes):
        y_true, y_pred, y_proba = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_proba)
        counts = confusion_matrix(y_true, y_pred, labels=classes)
        support = counts.sum(axis=1)
        present = support > 0
        recalls = np.divide(np.diag(counts), support, out=np.full(len(classes), np.nan), where=present)
        single_class = np.count_nonzero(present) == 1
        return {
            "n_windows": len(y_true),
            "support": {
                int(label): int(count)
                for label, count in zip(classes, support)
            },
            "recall": {
                int(label): float(recall) if np.isfinite(recall) else None
                for label, recall in zip(classes, recalls)
            },
            "single_class": bool(single_class),
            "f1_macro": (
                float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))
                if len(y_true) else None
            ),
            "accuracy": (
                float(accuracy_score(y_true, y_pred))
                if len(y_true) else None
            ),
            "balanced_accuracy": (
                float(balanced_accuracy_score(y_true, y_pred))
                if present.any() else None
            ),
            "mcc": (
                float(matthews_corrcoef(y_true, y_pred))
                if np.count_nonzero(present) >= 2 else None
            ),
            "auroc": (
                float(roc_auc_score(y_true, y_proba))
                if present.all() else None
            ),
            "confusion_matrix": counts.tolist(),
            "reason": (
                "empty test set" if not len(y_true)
                else "single true class: AUROC and MCC undefined" if single_class
                else None
            ),
        }

    def _get_clf(self):
        return Pipeline([("scaler", StandardScaler() if self.scale_embeddings else "passthrough"),
                        ("probe", LogisticRegression(solver="lbfgs", class_weight="balanced", max_iter=10000))])
