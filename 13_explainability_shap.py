"""
Step 13 — Fit the final model per antibiotic on all data, print its
human-readable risk formula, and generate SHAP explanations.

Logistic regression coefficients alone describe an AVERAGE effect across
the whole dataset; they don't explain a single prediction in isolation.
SHAP decomposes each individual prediction into an exact, additive
contribution from each of G/U/D/H, which is why this project can say
"predicted resistant mainly due to acquired-gene evidence, reinforced by
plasmid mobility" instead of just a bare probability. For a linear model
this is computed exactly by features.linear_shap(); this step also
cross-checks that against the `shap` library's own LinearExplainer to
confirm the two agree.
"""

import subprocess

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from config import FEATURES, FEATURE_MATRIX, ANTIBIOTIC_TO_SUBCLASS
from features import linear_shap
from sklearn.linear_model import LogisticRegression


def make_lr():
    return LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", random_state=0)


def fit_final_models(fm, platt):
    try:
        import shap
    except ImportError:
        subprocess.run(["pip", "install", "-q", "shap"])
        import shap

    final = {}
    for ab in ANTIBIOTIC_TO_SUBCLASS:
        d = fm[fm.antibiotic == ab].reset_index(drop=True)
        lr = make_lr().fit(d[FEATURES].values, d["phenotype"].values)
        final[ab] = dict(lr=lr, background=d[FEATURES].values.copy(), a=platt[ab][0], b=platt[ab][1])

        w, b0 = lr.coef_[0], lr.intercept_[0]
        rel = np.abs(w) / np.abs(w).sum()
        print(f"\n=== {ab} ===")
        print("  logit = " + f"{b0:+.2f} " + " ".join(f"{wi:+.2f}*{f}" for wi, f in zip(w, FEATURES)))
        print("  risk  = 1 / (1 + exp(-(%.2f * logit %+.2f)))   <- calibration step" % (platt[ab][0], platt[ab][1]))
        print("  relative weight |w|/sum|w| : " + "  ".join(f"{f}={r:.0%}" for f, r in zip(FEATURES, rel)))

        X = d[FEATURES].values
        sv = linear_shap(lr, X, X)
        try:
            ref = shap.LinearExplainer(lr, shap.maskers.Independent(X, max_samples=len(X))).shap_values(X)
            ref = ref[-1] if isinstance(ref, list) else ref
            print(f"  check against the SHAP library (LinearExplainer): max difference = {np.abs(ref - sv).max():.1e}")
        except Exception as e:
            print("  (SHAP-library cross-check skipped:", type(e).__name__, ")")

        shap.summary_plot(sv, d[FEATURES], show=False)
        plt.title(f"SHAP - {ab}")
        plt.show()

    return final


def main(platt):
    """`platt` is produced by modeling/12_train_models_cv.py's train_all()."""
    fm = pd.read_csv(FEATURE_MATRIX, sep="\t")
    return fit_final_models(fm, platt)


if __name__ == "__main__":
    import sys
    print("Run this after 12_train_models_cv.py, passing its `platt` dict to main(platt).")
    sys.exit(0)
