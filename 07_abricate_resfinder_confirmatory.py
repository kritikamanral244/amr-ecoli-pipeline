"""
Step 7 — ABRicate / ResFinder: confirmatory caller for Module G.

Relying on a single gene-detection tool means trusting that tool's calls
blindly. Running a second, independently-curated database (ResFinder, via
ABRicate) and checking which hits both tools agree on gives an internal
concordance check -- a basic cross-validation of the gene calls themselves.
Agreement becomes the `tool_concordance` signal used in step 08 to weight
how confidently a gene hit counts as evidence.

Run with a small thread pool (2 workers) since ABRicate is lightweight
enough that running genomes concurrently, rather than strictly
sequentially, is worthwhile even on a 2-core machine.
"""

import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import QC_TABLE, ABRICATE_RESFINDER_DIR
from common import qc_passed_fasta_files

N_WORKERS = 2


def run_abricate_resfinder(fasta_files, accession_of, n_workers=N_WORKERS):
    os.makedirs(ABRICATE_RESFINDER_DIR, exist_ok=True)

    def run_one(f):
        acc = accession_of[f]
        out_path = f"{ABRICATE_RESFINDER_DIR}/{acc}_resfinder.tsv"
        if os.path.exists(out_path):
            return acc, "skipped"
        t0 = time.time()
        result = subprocess.run(["abricate", "--db", "resfinder", f], capture_output=True, text=True)
        if result.returncode != 0:
            return acc, f"FAILED: {result.stderr[:300]}"
        with open(out_path, "w") as fh:
            fh.write(result.stdout)
        return acc, f"ok ({time.time() - t0:.0f}s)"

    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(run_one, f) for f in fasta_files]
        for fut in as_completed(futures):
            acc, status = fut.result()
            done += 1
            print(f"[{done}/{len(fasta_files)}] {acc}: {status}")
    print("ABRicate/ResFinder done")


def main():
    fasta_files, accession_of = qc_passed_fasta_files(QC_TABLE)
    run_abricate_resfinder(fasta_files, accession_of)


if __name__ == "__main__":
    main()
