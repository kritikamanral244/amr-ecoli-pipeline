"""
Step 6 — AMRFinderPlus: primary caller for Module G (acquired resistance
genes) and Module U (resistance-conferring point mutations).

AMRFinderPlus is the field-standard, curated tool for detecting both of
these evidence types in bacterial genomes, maintained directly by NCBI, and
is treated as the PRIMARY, AUTHORITATIVE caller in this pipeline -- its
hits are kept even when a confirmatory tool (step 07) disagrees. Run with
--plus (also screens stress/virulence genes) and --threads set to the full
core count. Printing elapsed time per genome makes an abnormally slow
genome visible as it runs rather than discovering it after the fact.
"""

import os
import subprocess
import time

from config import QC_TABLE, AMRFINDER_DIR
from common import qc_passed_fasta_files


def run_amrfinderplus(fasta_files, accession_of, n_threads=None):
    os.makedirs(AMRFINDER_DIR, exist_ok=True)
    n_threads = n_threads or os.cpu_count()
    print(f"Using {n_threads} threads")

    for f in fasta_files:
        acc = accession_of[f]
        out_path = f"{AMRFINDER_DIR}/{acc}.tsv"
        if os.path.exists(out_path):
            continue
        t0 = time.time()
        cmd = [
            "amrfinder", "-n", f, "-O", "Escherichia", "--plus",
            "--threads", str(n_threads), "-o", out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"AMRFinderPlus FAILED: {acc}\n{result.stderr[:400]}")
        else:
            print(f"{acc}: {elapsed:.1f}s")
    print("AMRFinderPlus done")


def main():
    fasta_files, accession_of = qc_passed_fasta_files(QC_TABLE)
    run_amrfinderplus(fasta_files, accession_of)


if __name__ == "__main__":
    main()
