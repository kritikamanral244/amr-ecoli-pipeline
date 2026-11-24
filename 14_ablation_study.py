"""
Step 14 — Ablation study: is each of G/U/D/H actually needed?

Building four separate evidence modules was significant extra engineering
(Prodigal, HMMER/Pfam, MOB-suite, PlasmidFinder) compared to a
gene-presence-only pipeline. This step is the direct, quantified test of
whether that extra effort was worthwhile: refit each antibiotic's model
four more times, each time with exactly one module removed, using the same
repeated grouped cross-validation as the full model (step 12), and report
the change in AUROC with a paired bootstrap confidence interval. A feature
only counts as "needed" if its whole 95% CI sits below zero (removing it
strictly hurt performance).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import FEATURES, FEATURE_MATRIX, ANTIBIOTIC_TO_SUBCLASS, N_BOOT, ABLATION_RESULTS
from cv_utils import make_lr, oof_scores, lr_score


def paired_boot_delta(y, s_full, s_drop, n=500, seed=1):
    rng, deltas = np.random.default_rng(seed), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        deltas.append(roc_auc_score(y[i], s_drop[i]) - roc_auc_score(y[i], s_full[i]))
    return np.percentile(deltas, [2.5, 97.5])


def run_ablation(fm):
    rows = []
    for ab in ANTIBIOTIC_TO_SUBCLASS:
        d = fm[fm.antibiotic == ab].reset_index(drop=True)
        y, g = d["phenotype"].values, d["split_group"].values

        s_full = oof_scores(d[FEATURES].values, y, g, make_lr, lr_score)
        auc_full = roc_auc_score(y, s_full)

        for f in FEATURES:
            keep = [c for c in FEATURES if c != f]
            s_drop = oof_scores(d[keep].values, y, g, make_lr, lr_score)
            lo, hi = paired_boot_delta(y, s_full, s_drop, n=N_BOOT // 2)
            delta = roc_auc_score(y, s_drop) - auc_full

            rows.append(dict(
                antibiotic=ab, dropped=f, AUROC_full=round(auc_full, 3),
                AUROC_without=round(roc_auc_score(y, s_drop), 3), delta=round(delta, 3),
                CI_95=f"[{lo:+.3f}, {hi:+.3f}]",
                verdict=(
                    "feature matters" if hi < 0 else
                    ("no clear effect (CI includes 0)" if lo <= 0 <= hi else "dropping it lost nothing")
                ),
            ))

    print("Same cross-validation for the full and each reduced model, so the difference is the feature alone.")
    print("A feature is only 'needed' if the whole CI is below 0.\n")
    ablation_df = pd.DataFrame(rows)
    print(ablation_df.to_string(index=False))

    ablation_df.to_csv(ABLATION_RESULTS, sep="\t", index=False)
    print(f"\nSaved: {ABLATION_RESULTS}")
    return ablation_df


def main():
    fm = pd.read_csv(FEATURE_MATRIX, sep="\t")
    return run_ablation(fm)


if __name__ == "__main__":
    main()
