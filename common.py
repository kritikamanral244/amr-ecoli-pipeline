"""
Small shared helpers used by more than one pipeline step (genome discovery,
safe table reading). Kept separate from config.py because these are
functions, not constants.
"""

import glob
import os

import pandas as pd

from config import GENOME_DIR


def list_fasta_files(genome_dir=GENOME_DIR):
    """Every downloaded genome FASTA, and a {path: accession} lookup."""
    fasta_files = sorted(glob.glob(f"{genome_dir}/*/*.fna"))
    accession_of = {f: os.path.basename(os.path.dirname(f)) for f in fasta_files}
    print(f"{len(fasta_files)} genomes found under {genome_dir}")
    return fasta_files, accession_of


def qc_passed_fasta_files(qc_table_path, genome_dir=GENOME_DIR):
    """Genome FASTA files, restricted to accessions that passed QC (step 03)."""
    fasta_files, accession_of = list_fasta_files(genome_dir)
    qc_df = pd.read_csv(qc_table_path, sep="\t")
    passed = set(qc_df.loc[qc_df["qc_pass"].astype(str).str.lower().isin(["true", "1"]), "assembly_accession"])
    fasta_files = [f for f in fasta_files if accession_of[f] in passed]
    print(f"{len(fasta_files)} genomes passed QC")
    return fasta_files, accession_of


def read_table_safe(path, sep="\t"):
    """pd.read_csv that returns an empty DataFrame instead of raising on a
    missing or empty file -- several tools legitimately produce no output
    for a genome with no hits."""
    try:
        return pd.read_csv(path, sep=sep, low_memory=False)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame()
