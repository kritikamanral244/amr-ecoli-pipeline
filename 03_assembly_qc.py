"""
Step 3 — Assembly quality control.

Computes total genome length, contig count, N50 (assembly contiguity), and
the percentage of ambiguous ("N") bases for every downloaded genome, and
flags pass/fail against pre-set thresholds. Running every downstream tool
(AMRFinderPlus, Prodigal, MOB-suite) on a broken or badly fragmented
assembly wastes real compute time and can silently produce misleading
results -- e.g. a resistance gene split across a contig break looking
absent. This is a "garbage in, garbage out" control that runs before any
annotation step.
"""

import pandas as pd
from Bio import SeqIO

from config import (
    QC_TABLE,
    QC_MIN_LENGTH,
    QC_MAX_LENGTH,
    QC_MIN_N50,
    QC_MAX_PCT_AMBIGUOUS,
    QC_MAX_CONTIGS,
)
from common import list_fasta_files


def assembly_stats(fasta_path):
    lengths = []
    n_count = 0
    total_len = 0
    for rec in SeqIO.parse(fasta_path, "fasta"):
        seq = str(rec.seq).upper()
        lengths.append(len(seq))
        n_count += seq.count("N")
        total_len += len(seq)
    lengths.sort(reverse=True)
    cum = 0
    n50 = 0
    for length in lengths:
        cum += length
        if cum >= total_len / 2:
            n50 = length
            break
    return {
        "total_length": total_len,
        "n_contigs": len(lengths),
        "n50": n50,
        "pct_ambiguous": (n_count / total_len * 100) if total_len else 100,
    }


def run_qc(fasta_files, accession_of):
    qc_rows = []
    for f in fasta_files:
        stats = assembly_stats(f)
        stats["assembly_accession"] = accession_of[f]
        qc_rows.append(stats)

    qc_df = pd.DataFrame(qc_rows)
    qc_df["qc_pass"] = (
        qc_df["total_length"].between(QC_MIN_LENGTH, QC_MAX_LENGTH)
        & (qc_df["n50"] >= QC_MIN_N50)
        & (qc_df["pct_ambiguous"] <= QC_MAX_PCT_AMBIGUOUS)
        & (qc_df["n_contigs"] <= QC_MAX_CONTIGS)
    )

    qc_df.to_csv(QC_TABLE, sep="\t", index=False)
    print(qc_df["qc_pass"].value_counts())
    print(f"Saved: {QC_TABLE}")
    return qc_df


def main():
    fasta_files, accession_of = list_fasta_files()
    qc_df = run_qc(fasta_files, accession_of)

    passed_accessions = set(qc_df.loc[qc_df["qc_pass"], "assembly_accession"])
    fasta_files = [f for f in fasta_files if accession_of[f] in passed_accessions]
    print(f"{len(fasta_files)} genomes passed QC and continue into the pipeline")
    return fasta_files, accession_of


if __name__ == "__main__":
    main()
