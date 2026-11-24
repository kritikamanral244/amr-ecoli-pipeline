"""
Step 4 — MLST (Multi-Locus Sequence Typing): assign each QC-passed genome a
Sequence Type (ST), a standard genetic fingerprint used to classify
bacterial lineages (e.g. ST131, a well-known globally distributed E. coli
lineage).

Isolates from the same outbreak or closely related lineage can be
genetically near-identical. If such isolates end up split between the
training and test sets, a model can effectively "recognize" the lineage
rather than learning genuine resistance biology -- producing misleadingly
high accuracy. The Sequence Type computed here becomes `split_group` in the
feature matrix (modeling/11_feature_matrix_build.py), used to build a
lineage-aware train/calibration/test split.
"""

import os
import subprocess

import pandas as pd

from config import QC_TABLE, MLST_TABLE
from common import qc_passed_fasta_files


def run_mlst(fasta_files, accession_of):
    mlst_out = subprocess.run(["mlst"] + fasta_files, capture_output=True, text=True)
    mlst_lines = [line.split("\t") for line in mlst_out.stdout.strip().split("\n") if line]

    if not mlst_lines:
        print("mlst produced no output")
        return pd.DataFrame(columns=["assembly_accession", "scheme", "ST"])

    columns = ["file", "scheme", "ST"] + [f"locus_{i}" for i in range(len(mlst_lines[0]) - 3)]
    mlst_df = pd.DataFrame(mlst_lines, columns=columns)
    mlst_df["assembly_accession"] = mlst_df["file"].apply(
        lambda p: accession_of.get(p, os.path.basename(os.path.dirname(p)))
    )

    out = mlst_df[["assembly_accession", "scheme", "ST"]]
    out.to_csv(MLST_TABLE, sep="\t", index=False)
    print(f"Saved {len(out)} MLST calls -> {MLST_TABLE}")
    print(f"Distinct sequence types: {out['ST'].nunique()}")
    return out


def main():
    fasta_files, accession_of = qc_passed_fasta_files(QC_TABLE)
    return run_mlst(fasta_files, accession_of)


if __name__ == "__main__":
    main()
