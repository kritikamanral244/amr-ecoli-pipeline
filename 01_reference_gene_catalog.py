"""
Step 1 — Build a verified gene -> antibiotic-class mapping from NCBI's own
AMRFinderPlus Reference Gene Catalog.

Why this exists: guessing which gene belongs to which antibiotic class by
hand is a common, silent source of error in AMR pipelines. This script
downloads NCBI's official catalog (every gene AMRFinderPlus can detect,
with its curated drug class/subclass), extracts every distinct gene family
that actually appears in your working dataset's genotype strings, and
cross-references the two -- so the mapping used later in the pipeline is
evidence-based, not assumed.

Run once, before anything else. Its output (gene_class_map_VERIFIED.csv) is
a reference artifact for manual review; the actual class filters used
downstream live in config.ANTIBIOTIC_TO_SUBCLASS.
"""

import os
import re
import subprocess

import pandas as pd

from config import WORK, AST_CSV

REF_CATALOG_DIR = f"{WORK}/amrfinder_db"
REF_CATALOG_PATH = f"{REF_CATALOG_DIR}/ReferenceGeneCatalog.txt"
VERIFIED_MAP_PATH = f"{WORK}/gene_class_map_VERIFIED.csv"


def download_reference_catalog():
    os.makedirs(REF_CATALOG_DIR, exist_ok=True)
    if os.path.exists(REF_CATALOG_PATH):
        return
    url = (
        "https://ftp.ncbi.nlm.nih.gov/pathogen/Antimicrobial_resistance/"
        "AMRFinderPlus/database/latest/ReferenceGeneCatalog.txt"
    )
    subprocess.run(["curl", "-L", "-o", REF_CATALOG_PATH, url], check=True)


def parse_genotype_string(s):
    """'blaCTX-M-55=COMPLETE,gyrA_D87N=POINT' -> ['blaCTX-M-55', 'gyrA_D87N']"""
    if pd.isna(s) or not str(s).strip():
        return []
    out = []
    for chunk in str(s).split(","):
        if "=" in chunk:
            det, _method = chunk.split("=", 1)
            out.append(det.strip())
    return out


def to_family(det):
    """Collapse an individual determinant call to its gene family.
    Point mutations like gyrA_S83L -> family 'gyrA'.
    Allele numbers are stripped, e.g. blaTEM-1 -> blaTEM."""
    if re.search(r"_[A-Za-z]\d+[A-Za-z*]", det):
        return det.split("_")[0]
    d2 = re.sub(r"-\d+$", "", det)
    d2 = re.sub(r"(?<=[A-Za-z])\d+$", "", d2)
    return d2


def main():
    download_reference_catalog()

    ref = pd.read_csv(REF_CATALOG_PATH, sep="\t")
    ref_amr = ref[ref["type"] == "AMR"].copy()

    gene_class_map = (
        ref_amr[["gene_family", "class", "subclass"]]
        .dropna(subset=["gene_family"])
        .drop_duplicates(subset=["gene_family"])
    )

    df = pd.read_csv(AST_CSV)
    isolates = df.drop_duplicates(subset=["biosample_accession"]).copy()

    all_genes = set()
    for s in isolates["amr_genotypes_core"].dropna():
        for det in parse_genotype_string(s):
            all_genes.add(to_family(det))
    print(f"{len(all_genes)} distinct gene families found in the working dataset")

    my_genes_df = pd.DataFrame({"gene_family": sorted(all_genes)})
    verified = my_genes_df.merge(gene_class_map, on="gene_family", how="left")
    verified.to_csv(VERIFIED_MAP_PATH, index=False)

    matched = verified["class"].notna().sum()
    print(f"Matched to a real NCBI class: {matched} / {len(verified)}")
    print(f"Saved: {VERIFIED_MAP_PATH}")

    print("\nReview this before locking config.ANTIBIOTIC_TO_SUBCLASS:")
    print(
        gene_class_map[
            gene_class_map["class"].str.contains("BETA-LACTAM|QUINOLONE", case=False, na=False)
        ][["class", "subclass"]].drop_duplicates().to_string(index=False)
    )


if __name__ == "__main__":
    main()
