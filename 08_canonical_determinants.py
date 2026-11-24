"""
Step 8 — Canonical determinant table: merge AMRFinderPlus and ABRicate
results into the single table every later module reads from.

Gene symbols from both tools are "canonicalized" (uppercased, allele-number
suffixes stripped) so that e.g. blaTEM-1 and blaTEM-52 are recognized as the
same gene family when cross-matching between tools -- without this, hits
that genuinely agree would be missed purely due to naming/formatting
differences. Each AMRFinderPlus hit is flagged with whether ABRicate
independently confirmed it (`tool_concordance`); AMRFinderPlus itself
remains the primary, authoritative evidence source (a valid AMRFinderPlus
hit is kept even when ABRicate did not report it).

Note: RGI/CARD was installed in the tool environment but is NOT used here
-- its output was never wired into this table in the working pipeline. See
README "Honest notes".
"""

import pandas as pd

from config import QC_TABLE, AMRFINDER_DIR, ABRICATE_RESFINDER_DIR, CANONICAL_DETERMINANTS
from common import qc_passed_fasta_files, read_table_safe


def canon(symbol):
    """Uppercase, strip allele-number suffix -- used only for the cross-tool match."""
    return str(symbol).upper().split("_")[0].split("-")[0]


def parse_amrfinder(acc):
    df = read_table_safe(f"{AMRFINDER_DIR}/{acc}.tsv")
    if df.empty:
        return df
    df["assembly_accession"] = acc
    df["tool"] = "amrfinderplus"
    return df


def parse_abricate(acc):
    path = f"{ABRICATE_RESFINDER_DIR}/{acc}_resfinder.tsv"
    df = read_table_safe(path)
    if df.empty:
        return df
    df["assembly_accession"] = acc
    df["tool"] = "abricate_resfinder"
    return df


def build_canonical_table(fasta_files, accession_of):
    amr_all, abr_all = [], []
    for f in fasta_files:
        acc = accession_of[f]
        amr_all.append(parse_amrfinder(acc))
        abr_all.append(parse_abricate(acc))

    amr_hits = pd.concat(amr_all, ignore_index=True) if amr_all else pd.DataFrame()
    abr_hits = pd.concat(abr_all, ignore_index=True) if abr_all else pd.DataFrame()

    print("AMRFinderPlus hits:", len(amr_hits), "| ABRicate hits:", len(abr_hits))
    print("Genomes with AMRFinderPlus hits:", amr_hits["assembly_accession"].nunique(), "of", len(fasta_files))

    gene_col = "Gene symbol" if "Gene symbol" in amr_hits.columns else "Element symbol"
    amr_hits["canonical_id"] = amr_hits[gene_col].apply(canon)

    abr_gene_col = "GENE" if "GENE" in abr_hits.columns else None
    if abr_gene_col:
        abr_hits["canonical_id"] = abr_hits[abr_gene_col].apply(canon)

    abr_keys = set(zip(abr_hits.get("assembly_accession", []), abr_hits.get("canonical_id", [])))
    amr_hits["confirmed_by_abricate"] = amr_hits.apply(
        lambda r: (r["assembly_accession"], r["canonical_id"]) in abr_keys, axis=1
    )
    amr_hits["tool_concordance"] = amr_hits["confirmed_by_abricate"].astype(int)

    canonical_determinants = amr_hits.copy()  # AMRFinderPlus stays the primary evidence source
    canonical_determinants.to_csv(CANONICAL_DETERMINANTS, sep="\t", index=False)
    print(f"Saved: {CANONICAL_DETERMINANTS}")
    return canonical_determinants


def main():
    fasta_files, accession_of = qc_passed_fasta_files(QC_TABLE)
    return build_canonical_table(fasta_files, accession_of)


if __name__ == "__main__":
    main()
