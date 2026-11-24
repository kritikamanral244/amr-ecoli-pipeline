"""
Step 10 — Module H: mobile-genetic context.

A resistance gene's location in the genome matters biologically, not just
its presence. A gene on the chromosome is inherited vertically and stays
put; the same gene on a plasmid -- especially a conjugative one -- can
transfer directly between different bacteria, even different species,
which is a major driver of how resistance spreads in a population.

Runs MOB-suite's mob_recon (in its own isolated conda environment -- see
environment_setup.sh for why) to reconstruct and classify each genome's
contigs as chromosome or plasmid, plus ABRicate against PlasmidFinder as a
secondary, faster replicon-typing check. Processes genomes in configurable
batches (default 50) purely as a practical accommodation for limited
session runtimes, not a scientific choice -- the result is identical to
running all genomes in one pass.
"""

import os
import subprocess

import pandas as pd

from config import QC_TABLE, MOBSUITE_DIR, ABRICATE_PLASMID_DIR, MOBSUITE_CONTIGS, MOBSUITE_CONDA_ENV
from common import qc_passed_fasta_files, read_table_safe

N_THREADS = 2
BATCH_SIZE = 50


def run_mobsuite_and_plasmidfinder(fasta_files, accession_of, batch_size=BATCH_SIZE, n_threads=N_THREADS):
    os.makedirs(MOBSUITE_DIR, exist_ok=True)
    os.makedirs(ABRICATE_PLASMID_DIR, exist_ok=True)

    for batch_start in range(0, len(fasta_files), batch_size):
        batch = fasta_files[batch_start:batch_start + batch_size]
        print(f"\n=== Batch {batch_start}-{batch_start + len(batch)} of {len(fasta_files)} ===")

        for i, f in enumerate(batch, 1):
            acc = accession_of[f]

            outdir = f"{MOBSUITE_DIR}/{acc}"
            if not os.path.exists(outdir):
                result = subprocess.run(
                    ["conda", "run", "-n", MOBSUITE_CONDA_ENV, "mob_recon",
                     "-i", f, "-o", outdir, "--force", "-n", str(n_threads)],
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    print(f"MOB-suite FAILED: {acc}\n{result.stderr[:500]}")
                else:
                    print(f"[{i}/{len(batch)}] MOB-suite complete: {acc}")
            else:
                print(f"[{i}/{len(batch)}] MOB-suite already done: {acc}")

            out_path = f"{ABRICATE_PLASMID_DIR}/{acc}.tsv"
            if not os.path.exists(out_path):
                result = subprocess.run(["abricate", "--db", "plasmidfinder", f], capture_output=True, text=True)
                if result.returncode == 0:
                    with open(out_path, "w") as fh:
                        fh.write(result.stdout)
                else:
                    print(f"PlasmidFinder FAILED: {acc}\n{result.stderr[:500]}")

    print("\nMOB-suite + PlasmidFinder done for all genomes")


def parse_mobsuite_contig_reports(fasta_files, accession_of):
    def parse_one(acc):
        path = f"{MOBSUITE_DIR}/{acc}/contig_report.txt"
        df = read_table_safe(path)
        if df.empty:
            return df
        df["assembly_accession"] = acc
        return df

    rows = [parse_one(accession_of[f]) for f in fasta_files]
    mob_hits = pd.concat([r for r in rows if not r.empty], ignore_index=True) if any(not r.empty for r in rows) else pd.DataFrame()
    mob_hits.to_csv(MOBSUITE_CONTIGS, sep="\t", index=False)
    print(f"Saved {len(mob_hits)} contig classifications -> {MOBSUITE_CONTIGS}")
    return mob_hits


def main():
    fasta_files, accession_of = qc_passed_fasta_files(QC_TABLE)
    run_mobsuite_and_plasmidfinder(fasta_files, accession_of)
    return parse_mobsuite_contig_reports(fasta_files, accession_of)


if __name__ == "__main__":
    main()
