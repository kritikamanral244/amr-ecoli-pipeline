"""
Step 9 — Module D: protein / domain confidence.

A gene can be detected as "present" by AMRFinderPlus while actually being
disrupted, truncated, or carrying a frameshift that destroys its function --
presence/absence alone cannot distinguish a working resistance gene from a
broken one.

Candidate proteins are selected by genomic COORDINATE OVERLAP between each
AMRFinderPlus/ABRicate hit and Prodigal's predicted gene boundaries (not
"any protein on the same contig", which would be far too permissive).
Every candidate protein, across every genome, is written into one combined
FASTA and searched against Pfam-A in a SINGLE hmmscan run (with live
progress and an ETA) -- much more efficient than one hmmscan invocation per
genome. Each resulting domain hit is classified complete / partial /
absent_or_truncated based on what fraction of the matched Pfam domain
model's length is covered by the alignment.
"""

import glob
import os
import re
import subprocess
import time

import pandas as pd

from config import (
    PRODIGAL_ROOT,
    PFAM_HMM,
    CANONICAL_DETERMINANTS,
    CANDIDATE_PROTEINS_FASTA,
    HMMSCAN_DOMTBL,
    DOMAIN_HITS,
)


def ensure_pfam_pressed():
    if not os.path.exists(PFAM_HMM + ".h3m"):
        print("Pressing Pfam-A (one-time)...")
        result = subprocess.run(["hmmpress", "-f", PFAM_HMM], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-1000:])


def acc_from_prodigal_path(path):
    match = re.match(r"(GC[AF]_\d+\.\d+)", os.path.basename(path))
    if match:
        return match.group(1)
    # /prodigal_out/ACCESSION/ACCESSION.faa
    return os.path.basename(os.path.dirname(path))


def read_fasta(path):
    name, seq = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq)
                name, seq = line[1:], []
            else:
                seq.append(line.strip())
    if name is not None:
        yield name, "".join(seq)


def build_candidate_protein_regions(canonical_determinants_path=CANONICAL_DETERMINANTS):
    """{assembly_accession: {contig: [(start, end), ...]}} from every AMR hit."""
    det = pd.read_csv(canonical_determinants_path, sep="\t", low_memory=False)
    det["assembly_accession"] = det["assembly_accession"].astype(str)

    def pick(names):
        return next((c for c in names if c in det.columns), None)

    c_contig = pick(["Contig id", "Contig_id", "contig_id"])
    c_start = pick(["Start", "start"])
    c_stop = pick(["Stop", "End", "stop", "end"])
    assert c_contig and c_start and c_stop, f"Need contig/start/stop columns; found {list(det.columns)}"

    det["_contig"] = det[c_contig].astype(str).str.split().str[0]
    det["_s"] = pd.to_numeric(det[c_start], errors="coerce")
    det["_e"] = pd.to_numeric(det[c_stop], errors="coerce")
    det = det.dropna(subset=["_s", "_e"])

    regions = {}
    for a, c, s, e in zip(det["assembly_accession"], det["_contig"], det["_s"], det["_e"]):
        regions.setdefault(a, {}).setdefault(c, []).append((min(s, e), max(s, e)))
    return regions


def write_candidate_protein_fasta(regions, prodigal_root=PRODIGAL_ROOT, out_path=CANDIDATE_PROTEINS_FASTA):
    faa_files = sorted(glob.glob(f"{prodigal_root}/**/*.faa", recursive=True))
    n_files, n_with_candidates, n_prot = 0, 0, 0
    unmatched = []

    with open(out_path, "w") as out:
        for faa in faa_files:
            n_files += 1
            acc = acc_from_prodigal_path(faa)
            if acc not in regions:
                unmatched.append(acc)
                continue

            got = 0
            for header, seq in read_fasta(faa):
                # Prodigal headers: '>contig_5 # start # end # strand # ID=...'
                parts = header.split(" # ")
                if len(parts) < 3:
                    continue
                pid = parts[0].split()[0]
                try:
                    g1, g2 = int(parts[1]), int(parts[2])
                except ValueError:
                    continue
                contig = pid.rsplit("_", 1)[0]

                if any(s <= max(g1, g2) and e >= min(g1, g2) for s, e in regions[acc].get(contig, [])):
                    out.write(f">{acc}|{pid}\n{seq.rstrip('*')}\n")
                    got += 1

            n_prot += got
            n_with_candidates += int(got > 0)

    print(f"Prodigal files: {n_files} | Genomes with candidate proteins: {n_with_candidates} | Proteins written: {n_prot}")
    if unmatched:
        print(f"{len(unmatched)} genomes had no AMR-hit accession match (examples: {unmatched[:3]})")
    assert n_prot > 0, "No candidate proteins found -- check accession/contig naming consistency"
    return out_path


def fmt_duration(sec):
    sec = int(sec)
    return f"{sec//3600:d}h {sec%3600//60:02d}m {sec%60:02d}s" if sec >= 3600 else f"{sec//60:d}m {sec%60:02d}s"


def run_hmmscan(candidate_fasta=CANDIDATE_PROTEINS_FASTA, domtbl_out=HMMSCAN_DOMTBL, cpu="2"):
    progress_out = f"{domtbl_out}.progress"

    with open(candidate_fasta) as f:
        total = sum(1 for line in f if line.startswith(">"))
    print(f"Sequences to scan: {total}")

    def done_count():
        try:
            with open(progress_out, "r", errors="ignore") as f:
                return sum(1 for line in f if line.startswith("Query:"))
        except FileNotFoundError:
            return 0

    cmd = ["hmmscan", "--cpu", cpu, "--cut_ga", "--noali", "-o", progress_out,
           "--domtblout", domtbl_out, PFAM_HMM, candidate_fasta]

    start = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while proc.poll() is None:
        time.sleep(5)
        n = done_count()
        elapsed = time.time() - start
        if n > 0:
            eta = elapsed / n * (total - n)
            print(f"\r[{n}/{total}] {n/total*100:5.1f}% | elapsed {fmt_duration(elapsed)} | ETA ~{fmt_duration(eta)}   ",
                  end="", flush=True)
        else:
            print(f"\rStarting up (loading Pfam)... elapsed {fmt_duration(elapsed)}   ", end="", flush=True)

    _out, err = proc.communicate()
    print(f"\n\nhmmscan exit code: {proc.returncode} | total time: {fmt_duration(time.time() - start)}")
    if proc.returncode != 0:
        print(err[-1500:])
    assert os.path.exists(domtbl_out), "hmmscan did not create the domain table"
    return domtbl_out


def parse_domain_table(domtbl_out=HMMSCAN_DOMTBL, out_path=DOMAIN_HITS):
    rows = []
    with open(domtbl_out) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            acc, pid = p[3].split("|", 1)
            rows.append({
                "target_domain": p[0], "query_protein": pid, "domain_len": float(p[2]),
                "ali_from": float(p[17]), "ali_to": float(p[18]),
                "hmm_from": float(p[15]), "hmm_to": float(p[16]),
                "assembly_accession": acc,
            })
    domain_hits = pd.DataFrame(rows)
    assert not domain_hits.empty, "hmmscan found no Pfam domains in the candidate proteins"

    domain_hits["hmm_coverage"] = (domain_hits["hmm_to"] - domain_hits["hmm_from"]) / domain_hits["domain_len"]
    domain_hits["domain_status"] = pd.cut(
        domain_hits["hmm_coverage"], bins=[-0.01, 0.5, 0.9, 1.01],
        labels=["absent_or_truncated", "partial", "complete"],
    )

    domain_hits.to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(domain_hits)} domain rows for {domain_hits.assembly_accession.nunique()} genomes -> {out_path}")
    print(domain_hits["domain_status"].value_counts(dropna=False))
    return domain_hits


def main():
    ensure_pfam_pressed()
    regions = build_candidate_protein_regions()
    write_candidate_protein_fasta(regions)
    run_hmmscan()
    return parse_domain_table()


if __name__ == "__main__":
    main()
