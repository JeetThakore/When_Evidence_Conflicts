"""
Driver for the instance-split vs question-split re-evaluation.

Runs run_conflict_prediction.py for every (split mode, seed) and, optionally,
the IC/ICC feature-set ablations, writing to results_repro/. Runs whose
output already exists are skipped, so the script can be resumed.

Usage
-----
  python run_split_comparison.py --seeds 0-19 --jobs 4
  python run_split_comparison.py --seeds 0-19 --jobs 4 --ablations
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODELS = ["llama3.1-8b", "meditron3-8b", "phi4-14b", "qwen3-4b", "qwen3-8b", "qwen3.5-9b"]
ABLATION_FEATURE_SETS = ["unc", "unc_docs", "question"]


def parse_seeds(spec: str) -> list[int]:
    seeds = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--modes", nargs="+", default=["instance", "question"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--ablations", action="store_true",
                    help="Also run IC/ICC feature-set ablations.")
    ap.add_argument("--repro-dir", default="results_repro")
    args = ap.parse_args()

    repro = Path(args.repro_dir)
    repro.mkdir(parents=True, exist_ok=True)
    embed_cache = repro / "embeddings.npz"
    jobs = []
    for seed in parse_seeds(args.seeds):
        for mode in args.modes:
            jobs.append((f"{mode}_seed{seed}", mode, seed, "full", None))
            if args.ablations:
                for fs in ABLATION_FEATURE_SETS:
                    jobs.append((f"ablation_{fs}_{mode}_seed{seed}", mode, seed, fs, ["IC", "ICC"]))

    def run(job):
        name, mode, seed, fs, conds = job
        out_dir = repro / name
        if all((out_dir / f"{m}_conflict_prediction.json").exists() for m in MODELS):
            return name, "skipped"
        cmd = [sys.executable, "run_conflict_prediction.py", "--models", *MODELS,
               "--out-dir", str(out_dir), "--skip-figures", "--split", mode,
               "--seed", str(seed), "--feature-set", fs, "--embed-cache", str(embed_cache)]
        if conds:
            cmd += ["--conditions", *conds]
        log = repro / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w", encoding="utf-8") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
        return name, f"exit {rc}"

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for name, status in pool.map(run, jobs):
            print(f"{name}: {status}", flush=True)


if __name__ == "__main__":
    main()
