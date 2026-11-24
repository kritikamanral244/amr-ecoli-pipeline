"""
Shared cross-validation helpers used by both 12_train_models_cv.py (main
training/evaluation) and 14_ablation_study.py (which needs the exact same
grouped, repeated out-of-fold scoring to make a fair G/U/D/H comparison).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from config import N_SPLITS, N_REPEATS


def make_lr():
    return LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", random_state=0)


def grouped_folds(groups, n_splits, seed):
    """Assign whole lineages (groups) to folds with greedy load-balancing,
    so fold sizes stay roughly equal without ever splitting a lineage."""
    rng = np.random.default_rng(seed)
    sizes = pd.Series(groups).value_counts()
    order = (sizes * (1 + 0.4 * rng.random(len(sizes)))).sort_values(ascending=False).index
    load, fold_of = np.zeros(n_splits), {}
    for g in order:
        k = int(np.argmin(load))
        fold_of[g] = k
        load[k] += sizes[g]
    return pd.Series(groups).map(fold_of).values


def oof_scores(X, y, groups, make_model, score_fn, n_splits=N_SPLITS, repeats=N_REPEATS):
    """Out-of-fold score for every genome, averaged over repeated random
    grouped splits -- every genome is scored only by models that never saw
    its lineage."""
    total = np.zeros(len(y))
    for r in range(repeats):
        fold = grouped_folds(groups, n_splits, seed=100 + r)
        s = np.full(len(y), np.nan)
        for k in range(n_splits):
            te, tr = fold == k, fold != k
            if te.sum() == 0 or len(np.unique(y[tr])) < 2:
                raise ValueError("a fold has only one class in training -- too few resistant/susceptible genomes")
            s[te] = score_fn(make_model().fit(X[tr], y[tr]), X[te])
        total += s
    return total / repeats


lr_score = lambda m, X: m.decision_function(X)
rf_score = lambda m, X: m.predict_proba(X)[:, 1]
