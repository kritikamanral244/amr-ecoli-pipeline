"""
Step 15 — Plain-language per-isolate reports, the AMR burden index, and the
saved model bundle used for inference on new genomes.

A model's predictions are only genuinely useful and auditable if they come
with their supporting evidence attached, not a bare probability. Every
report row carries the raw G/U/D/H values and a plain-language explanation
alongside the cross-validated probability (cross-validated: this genome's
own prediction never came from a model that was trained on it). The saved
`amr_model_bundle.joblib` is the single artifact predict/16_predict_fasta.py
loads to score a brand-new genome -- it carries the fitted model, the
calibration parameters, and the exact antibiotic/feature mapping used at
training time, so inference can never silently drift from training.
"""

import joblib
import pandas as pd

from config import (
    FEATURE_MATRIX, ANTIBIOTIC_TO_SUBCLASS, FEATURES, U_CAP,
    ISOLATE_REPORTS_CSV, ISOLATE_REPORTS_JSON, AMR_BURDEN_INDEX, MODEL_BUNDLE_PATH,
)
from features import contributions, explain_text, risk_band


def build_reports(fm, oof, final):
    records = []
    for ab in ANTIBIOTIC_TO_SUBCLASS:
        d = fm[fm.antibiotic == ab].reset_index(drop=True)
        for r in d.itertuples():
            x = [getattr(r, f) for f in FEATURES]
            p = float(oof[ab][r.assembly_accession])
            c = contributions(final[ab], x)
            records.append(dict(
                assembly_accession=r.assembly_accession, antibiotic=ab, true_phenotype=r.phenotype,
                cv_probability=round(p, 3), risk_band=risk_band(p),
                **{f: round(v, 2) for f, v in zip(FEATURES, x)},
                explanation=explain_text(ab, p, c, x),
                prediction_type="cross-validated (this genome was NOT used to fit its own score)",
            ))

    reports = pd.DataFrame(records)
    burden = reports.groupby("assembly_accession")["cv_probability"].mean().mul(100).round(1).rename("amr_burden_index_0_100")
    reports = reports.merge(burden, on="assembly_accession")

    reports.to_csv(ISOLATE_REPORTS_CSV, index=False)
    reports.to_json(ISOLATE_REPORTS_JSON, orient="records", indent=2)
    burden.reset_index().to_csv(AMR_BURDEN_INDEX, index=False)

    print(f"Saved: {ISOLATE_REPORTS_CSV}")
    print(f"Saved: {ISOLATE_REPORTS_JSON}")
    print(f"Saved: {AMR_BURDEN_INDEX}")
    return reports, burden


def save_model_bundle(final):
    bundle = dict(
        models=final, features=FEATURES, mapping=ANTIBIOTIC_TO_SUBCLASS, u_cap=U_CAP,
        note="E. coli only. Research use only - not a diagnostic.",
    )
    joblib.dump(bundle, MODEL_BUNDLE_PATH)
    print(f"Saved model bundle: {MODEL_BUNDLE_PATH}")


def print_examples(reports, burden):
    print("\nThree example genomes:")
    sorted_accs = reports.sort_values("amr_burden_index_0_100", ascending=False).assembly_accession.unique()
    for acc in sorted_accs[[0, len(burden) // 2, -1]]:
        print(f"\n{acc}  burden index = {burden[acc]}")
        for t in reports[reports.assembly_accession == acc].explanation:
            print("  -", t)


def main(fm, oof, final):
    """`fm` from modeling/11_feature_matrix_build.py, `oof`/`final` from
    modeling/12_train_models_cv.py and modeling/13_explainability_shap.py."""
    reports, burden = build_reports(fm, oof, final)
    save_model_bundle(final)
    print_examples(reports, burden)
    return reports, burden


if __name__ == "__main__":
    import sys
    print("Run this after 11/12/13, passing fm, oof and final to main(fm, oof, final).")
    sys.exit(0)
