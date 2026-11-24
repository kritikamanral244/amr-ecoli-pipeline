"""
Step 12 — Repeated, lineage-aware cross-validation: train one L2-regularised
logistic regression per antibiotic and report performance with 95%
bootstrap confidence intervals.

Whole lineages (MLST sequence types) go into a single fold, never split
across folds -- the same leakage control as the train/test split, applied
here to cross-validation itself. Repeating the grouped split 5 times with
different random fold assignments, and averaging, gives every genome an
out-of-fold prediction from a model that never saw its own lineage, using
close to the full dataset rather than one 20-40 genome holdout. A
majority-class baseline, a one-line "any gene/mutation present" rule, and
a random-forest comparator are evaluated alongside the logistic regression
so its performance can be judged against real alternatives, not assumed.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    brier_score_loss, confusion_matrix,
)
from sklearn.linear_model import LogisticRegression

from config import FEATURES, FEATURE_MATRIX, ANTIBIOTIC_TO_SUBCLASS, N_BOOT
from features import apply_platt
from cv_utils import make_lr, oof_scores, lr_score, rf_score


def fit_platt(score, y):
    c = LogisticRegression(C=1e6, solver="lbfgs").fit(score.reshape(-1, 1), y)
    return float(c.coef_[0, 0]), float(c.intercept_[0])


def metrics_from(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        AUROC=roc_auc_score(y, p), AUPRC=average_precision_score(y, p), MCC=matthews_corrcoef(y, pred),
        sensitivity=tp / (tp + fn) if tp + fn else np.nan,
        specificity=tn / (tn + fp) if tn + fp else np.nan,
        Brier=brier_score_loss(y, p),
    )


def bootstrap_ci(y, p, n=N_BOOT, seed=0):
    rng, vals = np.random.default_rng(seed), {"AUROC": [], "AUPRC": [], "MCC": []}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals["AUROC"].append(roc_auc_score(y[i], p[i]))
        vals["AUPRC"].append(average_precision_score(y[i], p[i]))
        vals["MCC"].append(matthews_corrcoef(y[i], (p[i] >= 0.5).astype(int)))
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in vals.items()}


def train_all(fm):
    fmt = lambda v, ci: f"{v:.2f} [{ci[0]:.2f}-{ci[1]:.2f}]"
    oof, platt, summary = {}, {}, []

    for ab in ANTIBIOTIC_TO_SUBCLASS:
        d = fm[fm.antibiotic == ab].reset_index(drop=True)
        X, y, g = d[FEATURES].values, d["phenotype"].values, d["split_group"].values

        s = oof_scores(X, y, g, make_lr, lr_score)
        a, b = fit_platt(s, y)
        platt[ab] = (a, b)
        p = apply_platt(s, a, b)
        oof[ab] = pd.Series(p, index=d["assembly_accession"])

        m, ci = metrics_from(y, p), bootstrap_ci(y, p)
        rf_p = oof_scores(X, y, g, lambda: RandomForestClassifier(200, class_weight="balanced", random_state=0), rf_score)
        rule = ((d.G > 0) | (d.U > 0)).astype(int).values

        summary.append({
            "antibiotic": ab, "n": len(y), "resistant": int(y.sum()),
            "AUROC": fmt(m["AUROC"], ci["AUROC"]), "AUPRC": fmt(m["AUPRC"], ci["AUPRC"]),
            "MCC": fmt(m["MCC"], ci["MCC"]), "sensitivity": round(m["sensitivity"], 2),
            "specificity": round(m["specificity"], 2), "Brier": round(m["Brier"], 3),
            "AUPRC_if_guessing": round(y.mean(), 2), "rule(any hit) MCC": round(matthews_corrcoef(y, rule), 2),
            "RandomForest AUROC": round(roc_auc_score(y, rf_p), 2),
        })

    from config import N_SPLITS, N_REPEATS
    print(f"Repeated grouped CV: {N_SPLITS} folds x {N_REPEATS} repeats; "
          "every genome is scored by models that never saw its lineage.")
    print("Values are [95% bootstrap CI]. These come from ~all genomes, not a single small test split.\n")
    print(pd.DataFrame(summary).set_index("antibiotic").T.to_string())

    return oof, platt, summary


def main():
    fm = pd.read_csv(FEATURE_MATRIX, sep="\t")
    return train_all(fm)


if __name__ == "__main__":
    main()
