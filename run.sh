#!/usr/bin/env bash
# Run any script in this repo with the right Python import path set up.
#
#   ./run.sh pipeline/03_assembly_qc.py
#   ./run.sh modeling/11_feature_matrix_build.py
#   ./run.sh predict/16_predict_fasta.py my_genome.fna
#
# config.py, pipeline/common.py and modeling/features.py are shared across
# several steps; this just makes sure every script can `import` them
# regardless of which subfolder it lives in.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):$(pwd)/pipeline:$(pwd)/modeling:$(pwd)/predict${PYTHONPATH:+:$PYTHONPATH}"
python3 "$@"
