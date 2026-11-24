# AMR-XAI E. coli Pipeline

Explainable, calibrated antibiotic-resistance risk prediction from *E. coli*
whole-genome assemblies, built on four evidence modules:

| Module | What it measures |
|---|---|
| **G** | Acquired resistance genes (AMRFinderPlus primary, ABRicate/ResFinder confirmatory) |
| **U** | Resistance-conferring point mutations |
| **D** | Protein/domain confidence (is the hit gene's protein actually intact?) |
| **H** | Mobile-genetic context (is the hit on a plasmid, and how mobile is it?) |

Each module produces a 0–1 score per isolate-antibiotic pair. Those four
scores feed an L2-regularised logistic regression — one model per
antibiotic — whose output is calibrated, explained with SHAP, and
stress-tested with an ablation study and comparator baselines.

This repo is reorganised from three working Colab notebooks (data
acquisition/annotation, model training, and deployment/inference) into one
ordered, de-duplicated pipeline. File upload/download and Google Drive
cells are removed throughout — this code assumes your genome FASTA files
are already on local disk.

## Pipeline order

Run scripts in numeric order. Each one reads the files the previous steps
wrote, under `$AMR_WORK_DIR` (defaults to `/content`, override with that
environment variable for a non-Colab machine).

```
config.py                                   shared paths, antibiotic/evidence mapping, constants
environment_setup.sh                        installs the full tool stack (run first, once)

pipeline/
  01_reference_gene_catalog.py              verified gene -> antibiotic-class mapping from NCBI
  02_antibiotic_eligibility.py              AST data -> long format, which antibiotics have enough data
  03_assembly_qc.py                         genome QC (length, N50, contigs, ambiguous bases)
  04_mlst_lineage_typing.py                 Sequence Type per genome, for lineage-aware splitting
  05_prodigal_gene_calling.py               ORF prediction + protein sequences (Module D input)
  06_amrfinderplus_module_gu.py             primary caller for Modules G and U
  07_abricate_resfinder_confirmatory.py     confirmatory caller for Module G
  08_canonical_determinants.py              merges 06+07 into one evidence table
  09_domain_scan_module_d.py                HMMER/Pfam-A domain-completeness scan
  10_mobsuite_module_h.py                   plasmid reconstruction + mobility typing

modeling/
  features.py                               THE shared feature-building logic (used by training AND inference)
  cv_utils.py                               shared cross-validation helpers
  11_feature_matrix_build.py                builds the G/U/D/H training matrix from pipeline/ outputs
  12_train_models_cv.py                     repeated lineage-aware CV, calibration, comparator baselines
  13_explainability_shap.py                 final model fit, risk-formula printout, SHAP
  14_ablation_study.py                      is each of G/U/D/H actually needed?
  15_reports_and_bundle.py                  per-isolate reports + saves amr_model_bundle.joblib

predict/
  16_predict_fasta.py                       score one NEW genome against the trained bundle
```

### Running it

```bash
# one-time environment setup
AMR_WORK_DIR=/content ./environment_setup.sh

# each step (adjust AMR_WORK_DIR if not using Colab's /content)
export AMR_WORK_DIR=/content
./run.sh pipeline/03_assembly_qc.py
./run.sh pipeline/04_mlst_lineage_typing.py
./run.sh pipeline/05_prodigal_gene_calling.py
./run.sh pipeline/06_amrfinderplus_module_gu.py
./run.sh pipeline/07_abricate_resfinder_confirmatory.py
./run.sh pipeline/08_canonical_determinants.py
./run.sh pipeline/09_domain_scan_module_d.py
./run.sh pipeline/10_mobsuite_module_h.py
./run.sh modeling/11_feature_matrix_build.py
./run.sh modeling/12_train_models_cv.py
# 13, 14 and 15 need variables (platt, fm, oof, final) produced by the
# previous steps -- run them from a Python/Jupyter session rather than as
# standalone scripts, e.g.:
#   from importlib import import_module
#   step11 = import_module("11_feature_matrix_build")  # see note below
#   fm = step11.main()
#   step12 = import_module("12_train_models_cv")
#   oof, platt, summary = step12.train_all(fm)
#   step13 = import_module("13_explainability_shap")
#   final = step13.fit_final_models(fm, platt)
#   step14 = import_module("14_ablation_study")
#   step14.run_ablation(fm)
#   step15 = import_module("15_reports_and_bundle")
#   step15.main(fm, oof, final)

# inference on a new genome, once amr_model_bundle.joblib exists
./run.sh predict/16_predict_fasta.py /path/to/new_genome.fna
```

Note on step numbering: Python module names can't start with a digit, so
`01_reference_gene_catalog.py` etc. are run as scripts (`python3
pipeline/01_....py`), not `import`ed by number. Steps 01–10 and 16 are
fully standalone scripts with their own `main()`. Steps 11–15 are designed
to be chained together (each one's output feeds the next), so they're best
run from one interactive session rather than as five separate process
invocations — see the snippet above.

## Prerequisites

- Genome FASTA files already downloaded, laid out as
  `$AMR_WORK_DIR/ecoli_wgs/ncbi_dataset/data/<accession>/<accession>*.fna`
  (this repo does not include genome acquisition — bring your own
  downloaded assemblies and your AST phenotype CSV)
- `pip install -r requirements.txt`
- `./environment_setup.sh` for the bioinformatics tool stack (conda/mamba,
  AMRFinderPlus, Prodigal, ABRicate, HMMER, MOB-suite)

## Honest notes on what the working pipeline actually does

Worth stating plainly in a methodology write-up or to a supervisor:

- **RGI/CARD was installed but is not used.** `environment_setup.sh`
  installs it (it was part of the original tool stack), but Module G's
  canonical determinant table (`pipeline/08_canonical_determinants.py`) is
  built from AMRFinderPlus (primary) + ABRicate/ResFinder (confirmatory)
  only. The `tool_concordance` flag therefore reflects agreement between
  those two tools, not three.
- **The ceftriaxone evidence mapping excludes BETA-LACTAM.** An earlier
  version of `config.ANTIBIOTIC_TO_SUBCLASS` included both `CEPHALOSPORIN`
  and `BETA-LACTAM` for ceftriaxone; `BETA-LACTAM` was removed because it
  matches the chromosomal *ampC* gene present in almost every *E. coli*,
  which was giving a high G score even for susceptible genomes. The
  mapping in `config.py` is the corrected version.
- **Module H uses a 4-tier mobility score**, not a simple plasmid/
  chromosome binary: chromosome = 0.0, plasmid/mobility-class-unresolved =
  0.5, mobilizable = 0.75, conjugative = 1.0 (`features.mobility_score`).
  This is a plasmid-location/mobility **proxy**, not proof of actual
  transfer between bacteria.
- **Genome acquisition is out of scope for this repo.** All NCBI Datasets
  download, Google Drive mount, and `files.upload`/`files.download` cells
  from the original notebooks were removed — this pipeline starts from
  genome FASTA files already present on disk.

## Model bundle contents (`amr_model_bundle.joblib`)

A dict with:
- `models`: `{antibiotic: {lr, background, a, b}}` — the fitted logistic
  regression, its SHAP background set, and its Platt calibration
  parameters, per antibiotic
- `features`: `["G", "U", "D", "H"]`
- `mapping`: the antibiotic → AMRFinderPlus-subclass evidence mapping used
  at training time
- `u_cap`: the mutation-count cap used to compute U

`predict/16_predict_fasta.py` loads this bundle and nothing else to decide
what counts as evidence for a new genome — so inference can never silently
drift from how the model was trained.

## Publishing this to GitHub

This session has no GitHub account connected, so push it yourself:

```bash
cd amr-xai-ecoli-pipeline
git init
git add .
git commit -m "AMR-XAI E. coli pipeline: G/U/D/H evidence modules, calibrated per-antibiotic models, SHAP, ablation"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

(Create the empty repo on GitHub first — on github.com, "New repository" —
then paste its URL into the `git remote add origin` line above.)
