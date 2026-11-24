"""
Step 2 — Load the AST (antibiotic susceptibility testing) CSV, reshape it to
long format (one row per isolate-antibiotic pair), and decide which
antibiotics have enough resistant AND susceptible isolates to be modelled.

A model trained on an antibiotic with only a handful of resistant examples
cannot learn anything reliable -- it overfits to noise. This gate runs
before any modelling begins, on the full working dataset.
"""

import pandas as pd

from config import AST_CSV, MIN_PER_CLASS


def build_long_format(ast_csv=AST_CSV, antibiotic_cols=("ceftriaxone", "ciprofloxacin", "gentamicin")):
    raw = pd.read_csv(ast_csv)
    print(f"Loaded {raw.shape[0]} isolates, {raw.shape[1]} columns from {ast_csv}")

    long_df = raw.melt(
        id_vars=["assembly_accession"],
        value_vars=list(antibiotic_cols),
        var_name="antibiotic",
        value_name="phenotype_binary",
    )
    long_df = long_df.dropna(subset=["phenotype_binary"]).copy()
    long_df["phenotype_binary"] = long_df["phenotype_binary"].astype(int)
    long_df["assembly_accession"] = long_df["assembly_accession"].astype(str)

    print(f"Usable isolate-antibiotic pairs: {len(long_df)}")
    print(long_df["phenotype_binary"].value_counts())
    return long_df


def eligible_antibiotics(long_df, min_per_class=MIN_PER_CLASS):
    counts = long_df.groupby("antibiotic")["phenotype_binary"].agg(n="count", n_resistant="sum")
    counts["n_susceptible"] = counts["n"] - counts["n_resistant"]
    counts = counts.sort_values("n", ascending=False)
    print(counts)

    eligible = counts[
        (counts["n_resistant"] >= min_per_class) & (counts["n_susceptible"] >= min_per_class)
    ]
    print("\nEligible antibiotics:\n", eligible)
    return list(eligible.index)


def main():
    long_df = build_long_format()
    selected = eligible_antibiotics(long_df)
    print("\nSelected antibiotics:", selected)

    keep_accessions = long_df.loc[long_df["antibiotic"].isin(selected), "assembly_accession"].unique()
    print(f"{len(keep_accessions)} unique assemblies needed for the selected antibiotics")
    return long_df, selected


if __name__ == "__main__":
    main()
