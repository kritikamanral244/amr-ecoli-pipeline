"""
Step 5 — Prodigal gene-calling: predict the location and translated protein
sequence of every gene (ORF) in each QC-passed genome, in "single" mode
(appropriate for an isolate genome rather than a mixed metagenomic sample).

Prodigal does ONLY gene-calling, with no downstream similarity-search
annotation against reference databases (that is what Prokka also does, and
what makes Prokka slow -- a full run of Prokka across 200 genomes took
multiple days; Prodigal does the same gene-calling step in minutes because
it skips the annotation step this pipeline never needed). The protein
sequences and coordinates produced here are exactly what Module D (domain
confidence, step 09) needs to find candidate proteins near each resistance
gene hit.
"""

import os
import subprocess

from config import QC_TABLE, PRODIGAL_ROOT
from common import qc_passed_fasta_files


def run_prodigal(fasta_files, accession_of):
    os.makedirs(PRODIGAL_ROOT, exist_ok=True)
    for f in fasta_files:
        acc = accession_of[f]
        outdir = f"{PRODIGAL_ROOT}/{acc}"
        os.makedirs(outdir, exist_ok=True)
        faa_path = f"{outdir}/{acc}.faa"
        gff_path = f"{outdir}/{acc}.gff"
        if os.path.exists(faa_path) and os.path.getsize(faa_path) > 0:
            continue
        cmd = [
            "prodigal", "-i", f, "-a", faa_path, "-f", "gff", "-o", gff_path,
            "-p", "single", "-q",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Prodigal FAILED: {acc}\n{result.stderr[:400]}")
    print("Prodigal gene-calling done")


def main():
    fasta_files, accession_of = qc_passed_fasta_files(QC_TABLE)
    run_prodigal(fasta_files, accession_of)


if __name__ == "__main__":
    main()
