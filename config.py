"""
Shared configuration for the AMR-XAI E. coli pipeline.

Every pipeline/modeling/predict script imports its paths and constants from
here, so there is exactly one place that defines what counts as evidence for
each antibiotic, how harsh the mutation cap is, and where intermediate files
live. Keeping this in one module (instead of redefining it in every script,
as the original notebooks did) is what prevents the three notebooks from
quietly drifting out of sync with each other.
"""

import os

# ---------------------------------------------------------------------------
# Working directories
# ---------------------------------------------------------------------------
# WORK is where every intermediate file (QC tables, AMRFinderPlus output,
# domain hits, the feature matrix, the trained model bundle, ...) is read
# from and written to. Override with the AMR_WORK_DIR environment variable
# if you are not running this on Colab's /content.
WORK = os.environ.get("AMR_WORK_DIR", "/content")

DBS_DIR = f"{WORK}/dbs"
PFAM_HMM = f"{DBS_DIR}/Pfam-A.hmm"

GENOME_DIR = f"{WORK}/ecoli_wgs/ncbi_dataset/data"   # one folder per assembly, containing its .fna
PRODIGAL_ROOT = f"{WORK}/prodigal_out"                # {accession}/{accession}.faa + .gff
AMRFINDER_DIR = f"{WORK}/amrfinder_out"
ABRICATE_RESFINDER_DIR = f"{WORK}/abricate_out"
ABRICATE_PLASMID_DIR = f"{WORK}/abricate_plasmid_out"
MOBSUITE_DIR = f"{WORK}/mobsuite_out"

# Intermediate / output tables
AST_CSV = f"{WORK}/amr_ecoli_working_set_200.csv"
QC_TABLE = f"{WORK}/qc_assembly.tsv"
MLST_TABLE = f"{WORK}/mlst.tsv"
CANONICAL_DETERMINANTS = f"{WORK}/canonical_determinants.tsv"
DOMAIN_HITS = f"{WORK}/domain_hits.tsv"
MOBSUITE_CONTIGS = f"{WORK}/mobsuite_contigs.tsv"
FEATURE_MATRIX = f"{WORK}/feature_matrix_long.tsv"
ABLATION_RESULTS = f"{WORK}/ablation_results.tsv"
AMR_BURDEN_INDEX = f"{WORK}/amr_burden_index.tsv"
ISOLATE_REPORTS_CSV = f"{WORK}/isolate_antibiotic_reports.csv"
ISOLATE_REPORTS_JSON = f"{WORK}/isolate_antibiotic_reports.json"
MODEL_BUNDLE_PATH = f"{WORK}/amr_model_bundle.joblib"

# The combined candidate-protein FASTA and HMMER domain table used by the
# Module D (domain confidence) scan.
CANDIDATE_PROTEINS_FASTA = f"{WORK}/candidate_proteins_all.faa"
HMMSCAN_DOMTBL = f"{WORK}/hmmer_domtbl_all.txt"

# Separate conda environment MOB-suite is installed into, kept isolated from
# the main tool environment because its dependency pins conflict with
# AMRFinderPlus/RGI if installed side by side.
MOBSUITE_CONDA_ENV = "mobsuite_env"

# ---------------------------------------------------------------------------
# Antibiotic panel and evidence mapping
# ---------------------------------------------------------------------------
# Which AMRFinderPlus "Subclass" values count as relevant evidence for each
# antibiotic. This is the FINAL, corrected mapping (ceftriaxone's
# "BETA-LACTAM" entry was deliberately removed -- it matches the chromosomal
# ampC gene present in almost every E. coli, which was giving G ~= 1 even for
# susceptible genomes). See README "Honest notes" for the earlier, wrong
# version this replaces.
ANTIBIOTIC_TO_SUBCLASS = {
    "ciprofloxacin": ["QUINOLONE"],
    "ceftriaxone": ["CEPHALOSPORIN"],
    "gentamicin": ["AMINOGLYCOSIDE"],
}

FEATURES = ["G", "U", "D", "H"]   # the four evidence-module scores every model is trained on
U_CAP = 3.0                        # number of resistance mutations that maps U to 1.0

# ---------------------------------------------------------------------------
# Assembly QC thresholds
# ---------------------------------------------------------------------------
QC_MIN_LENGTH = 4_000_000
QC_MAX_LENGTH = 6_500_000
QC_MIN_N50 = 20_000
QC_MAX_PCT_AMBIGUOUS = 1.0
QC_MAX_CONTIGS = 300

# ---------------------------------------------------------------------------
# Modelling settings
# ---------------------------------------------------------------------------
MIN_PER_CLASS = 25          # minimum resistant AND susceptible isolates for an antibiotic to be modelled
N_SPLITS = 5                 # grouped cross-validation folds
N_REPEATS = 5                # repeats of the grouped CV, each with a different random fold assignment
N_BOOT = 1000                 # bootstrap resamples for 95% confidence intervals
RANDOM_SEED = 20260826

# Risk bands used only for the plain-language report text, not for anything
# clinically validated.
def risk_band(p):
    return "LOW" if p < 0.30 else ("MODERATE" if p < 0.70 else "HIGH")
