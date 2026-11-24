#!/usr/bin/env bash
# ============================================================================
# Environment setup for the AMR-XAI E. coli pipeline.
#
# Installs the full bioinformatics tool stack via micromamba/conda. MOB-suite
# is installed into its OWN separate environment (mobsuite_env) because its
# dependency pins conflict with AMRFinderPlus/RGI if installed alongside them
# in the same environment -- this was a real failure mode in the original
# notebooks, not a precaution added after the fact.
#
# Genome acquisition (NCBI Datasets download) is intentionally NOT part of
# this script. This pipeline assumes you already have your genome FASTA
# files on disk under $AMR_WORK_DIR/ecoli_wgs/ncbi_dataset/data/<accession>/.
# ============================================================================
set -euo pipefail

: "${AMR_WORK_DIR:=/content}"
mkdir -p "$AMR_WORK_DIR"
cd "$AMR_WORK_DIR"

echo "=== Installing core AMR tool stack (amrfinder, prodigal, mlst, abricate, hmmer) ==="
mamba install -y -c bioconda -c conda-forge \
    ncbi-amrfinderplus \
    prodigal \
    mlst \
    abricate \
    hmmer

echo "=== Installing MOB-suite into its own isolated environment ==="
mamba create -y -n mobsuite_env -c bioconda -c conda-forge mob_suite

echo "=== Updating reference databases ==="
amrfinder -u
abricate --setupdb
conda run -n mobsuite_env mob_init

echo "=== Downloading and indexing the Pfam-A domain database (~1.5 GB) ==="
mkdir -p "$AMR_WORK_DIR/dbs"
cd "$AMR_WORK_DIR/dbs"
if [ ! -f Pfam-A.hmm ]; then
    wget -q https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz
    gunzip -f Pfam-A.hmm.gz
fi
if [ ! -f Pfam-A.hmm.h3m ]; then
    hmmpress Pfam-A.hmm
fi
cd "$AMR_WORK_DIR"

echo "=== Versions ==="
amrfinder --version
prodigal -v 2>&1 | head -1
mlst --version
abricate --list | head -1
conda run -n mobsuite_env mob_recon --version

echo "Environment ready."
