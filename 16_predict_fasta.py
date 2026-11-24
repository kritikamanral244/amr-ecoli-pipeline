"""
Step 16 — Score one new genome FASTA against the trained model bundle.

Runs the exact same tool chain as the training pipeline (Prodigal ->
AMRFinderPlus -> ABRicate/ResFinder -> HMMER/Pfam-A -> MOB-suite) on a
single new genome, builds its G/U/D/H features with the SAME
genome_features() function used to build the training matrix (features.py),
and scores it with the saved, calibrated model bundle. This is the
deployment counterpart to modeling/11-15: everything upstream of the model
itself must match training exactly, or a prediction on a new genome would
silently mean something different from what the model was trained on.

Usage:
    python 16_predict_fasta.py /path/to/genome.fna
    python 16_predict_fasta.py /path/to/genome.fna --bundle /path/to/amr_model_bundle.joblib
"""

import argparse
import json
import os
import re
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd

from config import PFAM_HMM, MODEL_BUNDLE_PATH, MOBSUITE_CONDA_ENV, WORK
from common import read_table_safe
from features import (
    canon, coverage_to_score, mob_scores_from_report, read_prodigal_coords,
    prep_det, genome_features, apply_platt, contributions, explain_text, risk_band,
)


def run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(f"'{cmd[0]}' is not installed -- run environment_setup.sh first")
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{r.stderr[-800:]}")
    return r


def fasta_stats(path):
    n, total = 0, 0
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                n += 1
            else:
                total += len(line.strip())
    return n, total


def parse_domtbl_scores(path):
    """hmmscan --domtblout -> {protein: best domain score}. Same coverage
    rule as training (pipeline/09_domain_scan_module_d.py)."""
    best = {}
    if not os.path.exists(path):
        return best
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            cov = (float(p[16]) - float(p[15])) / float(p[2])
            best[p[3]] = max(best.get(p[3], 0.0), coverage_to_score(cov))
    return best


def predict_fasta(fasta_path, workdir=None, threads=None, reuse=True, bundle_path=MODEL_BUNDLE_PATH, show=True):
    bundle = joblib.load(bundle_path)
    name = re.sub(r"\.(fna|fa|fasta)$", "", os.path.basename(fasta_path))
    workdir = workdir or f"{WORK}/predict_{name}"
    os.makedirs(workdir, exist_ok=True)
    threads = threads or (os.cpu_count() or 2)
    need = lambda p: not (reuse and os.path.exists(p) and os.path.getsize(p) > 0)

    n_contigs, total_len = fasta_stats(fasta_path)
    caveats = ["Model trained on E. coli only; research use only - not a diagnostic."]
    if not (3.8e6 <= total_len <= 6.5e6):
        caveats.append(f"Genome size {total_len/1e6:.2f} Mb is outside the usual E. coli range (3.8-6.5 Mb): check the species/assembly.")
    if n_contigs > 500:
        caveats.append(f"{n_contigs} contigs: a fragmented assembly can miss genes.")

    # 1) Prodigal (same flags as training; 'meta' mode only for very short sequences)
    faa = f"{workdir}/{name}.faa"
    if need(faa):
        run(["prodigal", "-i", fasta_path, "-a", faa, "-f", "gff", "-o", f"{workdir}/{name}.gff",
             "-p", "single" if total_len >= 100_000 else "meta", "-q"])

    # 2) AMRFinderPlus (same flags as training)
    amr_tsv = f"{workdir}/{name}_amrfinder.tsv"
    if need(amr_tsv):
        run(["amrfinder", "-n", fasta_path, "-O", "Escherichia", "--plus",
             "--threads", str(threads), "-o", amr_tsv])

    # 3) ABRicate/ResFinder -- the second tool used for the concordance check
    abr_tsv = f"{workdir}/{name}_resfinder.tsv"
    if need(abr_tsv):
        with open(abr_tsv, "w") as fh:
            fh.write(run(["abricate", "--db", "resfinder", fasta_path]).stdout)

    abr = read_table_safe(abr_tsv)
    abr_ids = set(abr["GENE"].map(canon)) if len(abr) and "GENE" in abr.columns else set()
    det = prep_det(read_table_safe(amr_tsv), abr_ids)
    prot = read_prodigal_coords(faa)

    # 4) Pfam domains, only for proteins that overlap an AMR hit
    cand_ids = set()
    for h in det.itertuples():
        cand_ids |= set(prot[(prot["contig"] == h.ctg) & (prot["start"] <= h.pos_stop) & (prot["end"] >= h.pos_start)]["pid"])
    cand_faa, dom_tbl = f"{workdir}/{name}_candidates.faa", f"{workdir}/{name}_domtbl.txt"
    if cand_ids and need(dom_tbl):
        keep, entries = False, []
        for line in open(faa):
            if line.startswith(">"):
                keep = line[1:].split()[0] in cand_ids
            if keep:
                entries.append(line.replace("*", "") if not line.startswith(">") else line)
        with open(cand_faa, "w") as f:
            f.write("".join(entries))
        run(["hmmscan", "--cpu", str(threads), "--cut_ga", "--noali", "-o", "/dev/null",
             "--domtblout", dom_tbl, PFAM_HMM, cand_faa])
    dom_score = parse_domtbl_scores(dom_tbl)

    # 5) MOB-suite: which contigs are plasmids (runs inside its own conda environment)
    mob_dir = f"{workdir}/mobsuite"
    report_txt = f"{mob_dir}/contig_report.txt"
    if need(report_txt):
        run(["conda", "run", "-n", MOBSUITE_CONDA_ENV, "mob_recon",
             "-i", fasta_path, "-o", mob_dir, "--force", "-n", str(threads)])
    mob_score = mob_scores_from_report(read_table_safe(report_txt))

    # 6) features -> saved model -> risk
    feats = genome_features(det, prot, dom_score, mob_score, bundle["mapping"], bundle["u_cap"])
    result = {"genome": name, "contigs": n_contigs, "length_Mb": round(total_len / 1e6, 2),
              "caveats": caveats, "antibiotics": {}}

    for ab, info in bundle["models"].items():
        x = [feats[ab][f] for f in bundle["features"]]
        z = info["lr"].decision_function(np.array([x]))[0]
        p = float(apply_platt(z, info["a"], info["b"]))
        c = contributions(info, x)
        result["antibiotics"][ab] = dict(
            probability=round(p, 3), band=risk_band(p), features=dict(zip(bundle["features"], x)),
            contributions={k: round(float(v), 2) for k, v in c.items()}, explanation=explain_text(ab, p, c, x),
        )

    result["amr_burden_index_0_100"] = round(100 * np.mean([v["probability"] for v in result["antibiotics"].values()]), 1)

    ev = det[det["Subclass"].isin(sum(bundle["mapping"].values(), []))]
    sym = "Gene symbol" if "Gene symbol" in ev.columns else "Element symbol"
    result["evidence"] = [
        dict(gene=r[sym], subclass=r["Subclass"], method=r["Method"], contig=r["ctg"],
             on_plasmid=mob_score.get(r["ctg"], 0.0) > 0,
             confirmed_by_second_tool=bool(r["tool_concordance"] >= 1))
        for r in ev.to_dict("records")
    ]

    with open(f"{workdir}/{name}_report.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    if show:
        print_report(result)
    return result


def print_report(r):
    print("=" * 70)
    print(f"AMR RISK REPORT - {r['genome']}   ({r['contigs']} contigs, {r['length_Mb']} Mb)")
    print("=" * 70)
    print(f"AMR burden index: {r['amr_burden_index_0_100']} / 100   (average predicted resistance chance over the drugs below)\n")
    for v in r["antibiotics"].values():
        print(" *", v["explanation"])
    if r["evidence"]:
        print("\nEvidence found:")
        for e in r["evidence"][:15]:
            print(f"   {e['gene']:<18} {e['subclass']:<16} {e['method']:<9} "
                  f"{'plasmid' if e['on_plasmid'] else 'chromosome':<11}"
                  f"{'2 tools agree' if e['confirmed_by_second_tool'] else '1 tool only'}")
    print("\nNotes:")
    for c in r["caveats"]:
        print("  -", c)


def main():
    parser = argparse.ArgumentParser(description="Score one genome FASTA against the trained AMR-XAI model bundle.")
    parser.add_argument("fasta_path", help="Path to the genome FASTA file (.fna/.fa/.fasta)")
    parser.add_argument("--bundle", default=MODEL_BUNDLE_PATH, help="Path to amr_model_bundle.joblib")
    parser.add_argument("--workdir", default=None, help="Working directory for intermediate tool output")
    parser.add_argument("--threads", type=int, default=None, help="Thread count (defaults to all CPU cores)")
    args = parser.parse_args()

    predict_fasta(args.fasta_path, workdir=args.workdir, threads=args.threads, bundle_path=args.bundle)


if __name__ == "__main__":
    main()
