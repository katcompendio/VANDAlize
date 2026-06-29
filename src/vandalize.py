"""
VANDAlize: shared analysis utilities.

Mechanistic interpretability of harm representations across English, Filipino,
and Taglish (Tagalog-English code-switching), using last-token residual-stream
activations.

Convention used throughout (matches the VANDA dataset CSVs): label 1 = harmful,
label 0 = benign. The harm direction is mean(harmful) - mean(benign), so
projections are higher for harmful prompts. Magnitudes (norms) of this raw
difference vector are the "signal strength" reported in the paper.

Note: the difference-of-means probe is sign-symmetric in the labels — flipping
the convention only flips the sign of the direction vector. Cosine magnitudes,
accuracy, AUROC, and direction norms are all identical either way.
"""

from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

SEED = 42


class DiffOfMeansProbe:
    """Difference-of-means linear probe (Marks & Tegmark, Geometry of Truth).

    direction      : unit-normalized (mean(benign) - mean(harmful))
    raw_direction  : un-normalized difference of means; its norm is the
                     magnitude / signal-strength used in the paper
    threshold      : midpoint between projected class means (LDA analog)
    """

    def __init__(self):
        self.direction = None
        self.raw_direction = None
        self.threshold = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        mu_harmful = X[y == 1].mean(axis=0)
        mu_benign = X[y == 0].mean(axis=0)
        raw = mu_harmful - mu_benign
        self.raw_direction = raw
        self.direction = raw / (np.linalg.norm(raw) + 1e-12)
        proj = X @ self.direction
        self.threshold = 0.5 * (proj[y == 1].mean() + proj[y == 0].mean())
        return self

    def decision_function(self, X):
        """Signed distance from the decision boundary (for AUROC).

        Higher = more harmful, since direction = mean(harmful) - mean(benign).
        """
        X = np.asarray(X, dtype=np.float32)
        return X @ self.direction - self.threshold

    def predict(self, X):
        """Binary prediction: 1 (harmful) if projection above threshold else 0."""
        return (self.decision_function(X) > 0).astype(int)


def harm_direction_norm(X, y):
    """Magnitude of the raw harm direction = signal strength."""
    probe = DiffOfMeansProbe().fit(X, y)
    return float(np.linalg.norm(probe.raw_direction))


def layer_sweep(layer_acts, y, test_size=0.2, seed=SEED):
    """Train/test accuracy of a difference-of-means probe at every layer.

    Args:
        layer_acts : dict {layer_idx: array [n, hidden]}
        y          : array [n] of labels
    Returns:
        (layers, train_acc, test_acc) as numpy arrays
    """
    layers = sorted(layer_acts.keys())
    y = np.asarray(y)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    train_acc, test_acc = [], []
    for L in layers:
        X = np.asarray(layer_acts[L])
        probe = DiffOfMeansProbe().fit(X[tr], y[tr])
        train_acc.append(accuracy_score(y[tr], probe.predict(X[tr])))
        test_acc.append(accuracy_score(y[te], probe.predict(X[te])))
    return np.array(layers), np.array(train_acc), np.array(test_acc)


def cosine_matrix(acts, labels, conditions, layer=14):
    """Pairwise cosine similarity of unit harm directions across conditions."""
    directions = {}
    for c in conditions:
        probe = DiffOfMeansProbe().fit(np.asarray(acts[c][layer]), np.asarray(labels[c]))
        directions[c] = probe.direction
    n = len(conditions)
    C = np.zeros((n, n))
    for i, a in enumerate(conditions):
        for j, b in enumerate(conditions):
            C[i, j] = float(np.dot(directions[a], directions[b]))
    return C, directions


def transfer_matrix(acts, labels, conditions, layer=14):
    """Cross-condition transfer: train probe on X, evaluate on Y, all pairs.

    Returns (accuracy_matrix, auroc_matrix), each [len(conditions) x len(conditions)],
    rows = train condition, cols = test condition.
    """
    n = len(conditions)
    M_acc = np.zeros((n, n))
    M_auc = np.zeros((n, n))
    for i, src in enumerate(conditions):
        probe = DiffOfMeansProbe().fit(np.asarray(acts[src][layer]), np.asarray(labels[src]))
        for j, tgt in enumerate(conditions):
            X = np.asarray(acts[tgt][layer])
            y = np.asarray(labels[tgt])
            M_acc[i, j] = accuracy_score(y, probe.predict(X))
            M_auc[i, j] = roc_auc_score(y, probe.decision_function(X))
    return M_acc, M_auc


def bootstrap_cosine(acts, labels, cond_a, cond_b, layer=14, n_boot=1000, seed=SEED):
    """Bootstrap 95% CI for cosine(harm_dir_a, harm_dir_b).

    Resamples prompts with replacement within each condition, recomputes the
    difference-of-means direction, and reports the spread of pairwise cosines.
    Returns (point_estimate, ci_low, ci_high, all_bootstrap_cosines).
    """
    rng = np.random.default_rng(seed)
    Xa, ya = np.asarray(acts[cond_a][layer]), np.asarray(labels[cond_a])
    Xb, yb = np.asarray(acts[cond_b][layer]), np.asarray(labels[cond_b])
    na, nb = len(ya), len(yb)
    cosines = []
    for _ in range(n_boot):
        ia = rng.integers(0, na, na)
        ib = rng.integers(0, nb, nb)
        if len(np.unique(ya[ia])) < 2 or len(np.unique(yb[ib])) < 2:
            continue
        da = DiffOfMeansProbe().fit(Xa[ia], ya[ia]).direction
        db = DiffOfMeansProbe().fit(Xb[ib], yb[ib]).direction
        cosines.append(float(np.dot(da, db)))
    cosines = np.array(cosines)
    point = float(np.dot(
        DiffOfMeansProbe().fit(Xa, ya).direction,
        DiffOfMeansProbe().fit(Xb, yb).direction,
    ))
    lo, hi = np.percentile(cosines, [2.5, 97.5])
    return point, float(lo), float(hi), cosines
