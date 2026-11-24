"""
The single, shared definition of the G/U/D/H feature-building logic, and of
how a fitted model's log-odds output is turned into a calibrated
probability and a plain-language explanation.

This module is imported by BOTH modeling/11_feature_matrix_build.py
(building the training feature matrix from saved pipeline outputs) and
predict/16_predict_fasta.py (scoring one new genome at inference time).
Keeping this logic in one place -- rather than copy-pasted into both the
training notebook and the deployment notebook, as the original two
notebooks did -- is what guarantees a genome is scored at inference time
using EXACTLY the same evidence rules it was trained with.

  G = acquired-gene evidence
  U = resistance-mutation evidence
  D = does the hit protein really carry its expected Pfam domain, intact?
  H = is the hit on a plasmid, and how mobile is that plasmid?
"""

import numpy as np
import pandas as pd

from config import ANTIBIOTIC_TO_SUBCLASS, U_CAP, FEATURES, risk_band

STATUS_SCORE = {"complete": 1.0, "partial": 0.5, "absent_or_truncated": 0.0}

LABELS = {
    "G": "acquired resistance genes",
    "U": "resistance-associated mutations",
    "D": "domain-confirmed resistance protein",
    "H": "plasmid location of the resistance gene",
}


def first_token(x):
    return str(x).split()[0] if pd.notna(x) and str(x).strip() else ""


def canon(symbol):
    """Same canonicalisation used in pipeline/08_canonical_determinants.py
    to compare AMRFinderPlus hits against ABRicate hits."""
    return str(symbol).upper().split("_")[0].split("-")[0]


def coverage_to_score(cov):
    return 1.0 if cov > 0.9 else (0.5 if cov > 0.5 else 0.0)


def mobility_score(molecule_type, predicted_mobility):
    """MOB-suite contig_report -> a 0..1 mobility score.
        chromosome                              -> 0.0
        plasmid, non-mobilizable                -> 0.5
        plasmid, mobility class unresolved       -> 0.5
        plasmid, mobilizable                     -> 0.75
        plasmid, conjugative                     -> 1.0
    This is a plasmid-location/mobility PROXY, not proof of actual transfer.
    """
    mol, pm = str(molecule_type).lower(), str(predicted_mobility).lower()
    if "chrom" in mol:
        return 0.0
    if "non" in pm:               # check "non-mobilizable" before "mobiliz"
        return 0.50
    if "conjug" in pm:
        return 1.00
    if "mobiliz" in pm:
        return 0.75
    if "plasmid" in mol:
        return 0.50
    return 0.0


def mob_scores_from_report(mob):
    """MOB-suite contig_report DataFrame (one genome) -> {contig: score}."""
    if mob is None or len(mob) == 0 or "contig_id" not in mob.columns:
        return {}
    mol = mob["molecule_type"] if "molecule_type" in mob.columns else pd.Series("", index=mob.index)
    pm = mob["predicted_mobility"] if "predicted_mobility" in mob.columns else pd.Series("", index=mob.index)
    return {
        first_token(c): mobility_score(a, b)
        for c, a, b in zip(mob["contig_id"], mol.fillna(""), pm.fillna(""))
    }


def read_prodigal_coords(faa_path):
    """Prodigal header: '>contig_5 # 337 # 2799 # 1 # ID=...'
    -> DataFrame(pid, contig, start, end)."""
    rows = []
    with open(faa_path) as f:
        for line in f:
            if line.startswith(">"):
                parts = line[1:].rstrip("\n").split(" # ")
                if len(parts) >= 3:
                    pid = parts[0].split()[0]
                    rows.append((pid, pid.rsplit("_", 1)[0], int(parts[1]), int(parts[2])))
    return pd.DataFrame(rows, columns=["pid", "contig", "start", "end"])


def prep_det(det, abr_ids=None):
    """Make a raw AMRFinderPlus table ready for feature building: adds
    ctg/pos_start/pos_stop/tool_concordance, handling either the on-disk
    canonical_determinants.tsv schema or a freshly-run amrfinder TSV."""
    det = det.copy()
    if len(det) == 0:
        for c in ("ctg", "pos_start", "pos_stop", "tool_concordance", "Subclass", "Method"):
            det[c] = pd.Series(dtype=object)
        return det

    cc = next(c for c in ("Contig id", "Contig_id", "contig_id") if c in det.columns)
    cs = next(c for c in ("Start", "start") if c in det.columns)
    ce = next(c for c in ("Stop", "End", "stop", "end") if c in det.columns)
    det["ctg"] = det[cc].map(first_token)
    s, e = pd.to_numeric(det[cs], errors="coerce"), pd.to_numeric(det[ce], errors="coerce")
    det["pos_start"], det["pos_stop"] = np.minimum(s, e), np.maximum(s, e)

    if "tool_concordance" not in det.columns:
        sym = "Gene symbol" if "Gene symbol" in det.columns else "Element symbol"
        det["tool_concordance"] = det[sym].map(lambda g: int(canon(g) in (abr_ids or set())))
    det["tool_concordance"] = pd.to_numeric(det["tool_concordance"], errors="coerce").fillna(0)

    return det.dropna(subset=["pos_start", "pos_stop"])


def genome_features(det, prot, dom_score, mob_score, mapping=None, u_cap=None):
    """Build the G/U/D/H scores for ONE genome, for every antibiotic in
    `mapping`.

    det        : prepped AMRFinderPlus hits for this genome (from prep_det)
    prot       : Prodigal protein coordinates for this genome (from read_prodigal_coords)
    dom_score  : {protein_id: 0..1 domain-completeness score}
    mob_score  : {contig_id: 0..1 mobility score}
    """
    mapping = mapping or ANTIBIOTIC_TO_SUBCLASS
    u_cap = u_cap or U_CAP
    out = {}

    for ab, subclasses in mapping.items():
        hits = det[det["Subclass"].isin(subclasses)] if len(det) else det
        is_point = (
            hits["Method"].astype(str).str.contains("POINT", na=False)
            if len(hits) else pd.Series([], dtype=bool)
        )
        gene_hits, mut_hits = hits[~is_point], hits[is_point]

        # G: capped sum of gene hits (confirmed by a second tool = 1.0, otherwise 0.8)
        G = min(1.0, sum(1.0 if c >= 1 else 0.8 for c in gene_hits["tool_concordance"])) if len(gene_hits) else 0.0

        # U: capped count of resistance mutations
        U = min(1.0, len(mut_hits) / u_cap)

        # D: for each hit, find proteins overlapping it by coordinates; each
        # protein is scored by its best Pfam domain (no Pfam hit -> 0).
        # D = mean over those proteins.
        scores = []
        for h in hits.itertuples():
            ov = prot[(prot["contig"] == h.ctg) & (prot["start"] <= h.pos_stop) & (prot["end"] >= h.pos_start)]
            scores += [dom_score.get(pid, 0.0) for pid in ov["pid"]]
        D = float(np.mean(scores)) if scores else 0.0

        # H: highest plasmid/mobility score among this antibiotic's hits
        H = max([mob_score.get(c, 0.0) for c in hits["ctg"]], default=0.0) if len(hits) else 0.0

        out[ab] = dict(G=G, U=U, D=D, H=H)

    return out


def linear_shap(lr, X, background):
    """SHAP values of a logistic-regression log-odds output, for
    independent features, are exactly weight * (value - average training
    value). This matches shap.LinearExplainer exactly (verified in
    modeling/13_explainability_shap.py) without requiring the shap package
    for this core computation."""
    return (np.asarray(X, dtype=float) - np.asarray(background, dtype=float).mean(axis=0)) * lr.coef_[0]


def apply_platt(score, a, b):
    """Platt/sigmoid calibration: raw log-odds -> calibrated probability."""
    return 1 / (1 + np.exp(-(a * np.asarray(score) + b)))


def contributions(model_info, x):
    """SHAP values (exact, for a linear model) for one genome's feature
    vector, as {feature: log-odds contribution}."""
    return dict(zip(FEATURES, linear_shap(model_info["lr"], [x], model_info["background"])[0]))


def explain_text(ab, p, contrib, x):
    """Plain-language, one-line explanation of a single antibiotic
    prediction, naming which evidence modules pushed the risk up or down."""
    ups = [f for f, v in sorted(contrib.items(), key=lambda kv: -kv[1]) if v > 0.05][:2]
    downs = [f for f, v in sorted(contrib.items(), key=lambda kv: kv[1]) if v < -0.05][:1]
    if x[0] == 0 and x[1] == 0:
        why = "No resistance gene or mutation for this drug was found."
    else:
        why = ("Raised by: " + ", ".join(LABELS[f] for f in ups) + ".") if ups else "No single dominant driver."
        if downs:
            why += " Lowered by: " + ", ".join(LABELS[f] for f in downs) + "."
    return f"{ab.capitalize()}: {risk_band(p)} risk ({p:.0%}). {why}"
