"""
Step 11 — Build the training feature matrix: one row per
isolate-antibiotic pair, with its G/U/D/H scores and its lineage
`split_group`.

This is where every module built in pipeline/03-10 (QC, MLST, Prodigal,
AMRFinderPlus, ABRicate, domain scan, mobility typing) converges into one
common, comparable numeric table, using the SAME genome_features() function
(modeling/features.py) that will later score new genomes at inference
time. Reducing all of that tool output down to four interpretable 0-1
scores per isolate-antibiotic pair is also what makes the SHAP
explanations in step 13 meaningful to a human reader.
"""

import re

import pandas as pd

from config import (
    WORK,
    AST_CSV,
    QC_TABLE,
    MLST_TABLE,
    CANONICAL_DETERMINANTS,
    DOMAIN_HITS,
    PRODIGAL_ROOT,
    ANTIBIOTIC_TO_SUBCLASS,
    FEATURE_MATRIX,
)
from features import STATUS_SCORE, mob_scores_from_report, read_prodigal_coords, prep_det, genome_features


def acc_from_name(path):
    m = re.search(r"(GC[AF]_\d+\.\d+)", path)
    return m.group(1) if m else path


def load_inputs():
    det_all = pd.read_csv(CANONICAL_DETERMINANTS, sep="\t", low_memory=False)
    dom_all = pd.read_csv(DOMAIN_HITS, sep="\t", low_memory=False)
    mob_all = pd.read_csv(f"{WORK}/mobsuite_contigs.tsv", sep="\t", low_memory=False)
    mlst = pd.read_csv(MLST_TABLE, sep="\t")
    qc = pd.read_csv(QC_TABLE, sep="\t")
    ast = pd.read_csv(AST_CSV)

    for d in (det_all, dom_all, mob_all, mlst, qc, ast):
        if "assembly_accession" in d.columns:
            d["assembly_accession"] = d["assembly_accession"].astype(str)

    return det_all, dom_all, mob_all, mlst, qc, ast


def map_phenotype(x):
    s = str(x).strip().lower()
    if s in {"r", "resistant", "1", "1.0"}:
        return 1
    if s in {"s", "susceptible", "sensitive", "0", "0.0"}:
        return 0
    return None


def build_long_phenotypes(ast, qc):
    passed = set(qc.loc[qc["qc_pass"].astype(str).str.lower().isin(["true", "1"]), "assembly_accession"])

    lower = {c.lower(): c for c in ast.columns}
    ab_cols = [lower[a] for a in ANTIBIOTIC_TO_SUBCLASS if a in lower]

    long_df = ast.melt(id_vars=["assembly_accession"], value_vars=ab_cols, var_name="antibiotic", value_name="raw")
    long_df["antibiotic"] = long_df["antibiotic"].str.lower()
    long_df["phenotype"] = long_df["raw"].map(map_phenotype)
    long_df = long_df.dropna(subset=["phenotype"])
    long_df["phenotype"] = long_df["phenotype"].astype(int)
    long_df = long_df[long_df["assembly_accession"].isin(passed)]
    return long_df


def build_feature_matrix(long_df, det_all, dom_all, mob_all, mlst):
    prodigal_files = {}
    import glob
    for p in glob.glob(f"{PRODIGAL_ROOT}/**/*.faa", recursive=True):
        prodigal_files.setdefault(acc_from_name(p), p)

    dom_all = dom_all.copy()
    dom_all["_score"] = dom_all["domain_status"].astype(str).map(STATUS_SCORE)
    dom_best = dom_all.dropna(subset=["_score"]).groupby(["assembly_accession", "query_protein"])["_score"].max()
    dom_by_acc = {a: g.droplevel(0).to_dict() for a, g in dom_best.groupby(level=0)}

    mob_by_acc = {a: mob_scores_from_report(g) for a, g in mob_all.groupby("assembly_accession")}
    det_by_acc = {a: g for a, g in det_all.groupby("assembly_accession")}
    st_by_acc = mlst.drop_duplicates("assembly_accession").set_index("assembly_accession")["ST"].astype(str).to_dict()

    rows, no_faa = [], 0
    for acc in sorted(long_df["assembly_accession"].unique()):
        det = prep_det(det_by_acc.get(acc, det_all.iloc[0:0]))
        if acc in prodigal_files:
            prot = read_prodigal_coords(prodigal_files[acc])
        else:
            prot = pd.DataFrame(columns=["pid", "contig", "start", "end"])
            no_faa += 1

        feats = genome_features(det, prot, dom_by_acc.get(acc, {}), mob_by_acc.get(acc, {}))
        st = st_by_acc.get(acc, "-")
        group = st if st not in ("-", "nan", "", "None") else f"untyped_{acc}"  # untyped genomes are their own group

        for r in long_df[long_df["assembly_accession"] == acc].itertuples():
            rows.append(dict(
                assembly_accession=acc, antibiotic=r.antibiotic, phenotype=r.phenotype,
                **feats[r.antibiotic], split_group=group,
            ))

    fm = pd.DataFrame(rows)
    fm.to_csv(FEATURE_MATRIX, sep="\t", index=False)

    print(f"{fm.assembly_accession.nunique()} genomes, {len(fm)} genome x antibiotic rows")
    print(f"genomes with no Prodigal .faa found (D = 0 for them): {no_faa}")
    print(f"lineage groups (sequence types): {fm.split_group.nunique()} | "
          f"largest group: {fm.drop_duplicates('assembly_accession').split_group.value_counts().iloc[0]} genomes")
    print(f"Saved: {FEATURE_MATRIX}")
    return fm


def sanity_checks(fm, det_all):
    print("\nSubclass values per antibiotic CLASS in the hit table (cross-check against config.ANTIBIOTIC_TO_SUBCLASS):")
    if "Class" in det_all.columns:
        print(det_all.groupby(["Class", "Subclass"]).size().sort_values(ascending=False).head(25).to_string())

    print("\nDoes the mapping find evidence where it should? (share of genomes with G>0 or U>0)")
    chk = fm.assign(has_evidence=(fm.G > 0) | (fm.U > 0)).groupby(["antibiotic", "phenotype"])["has_evidence"].mean().unstack().round(2)
    chk.columns = ["susceptible", "resistant"]
    print(chk.to_string())
    print("(want: resistant column high, susceptible column low. If both are high, or resistant is low, fix config.ANTIBIOTIC_TO_SUBCLASS.)")

    print("\nFeature correlation (D can only be >0 when there is a hit, so expect it to track G/U):")
    from config import FEATURES
    print(fm[FEATURES].corr().round(2).to_string())


def main():
    det_all, dom_all, mob_all, mlst, qc, ast = load_inputs()
    long_df = build_long_phenotypes(ast, qc)
    fm = build_feature_matrix(long_df, det_all, dom_all, mob_all, mlst)
    sanity_checks(fm, det_all)
    return fm


if __name__ == "__main__":
    main()
