#!/usr/bin/env python3
"""
run_all_commands.py
Sequentially run a set of feature-selection experiments with different methods
and settings. If any command fails, the script logs the failure and continues.

Usage:
    python run_all_commands.py
(Optional) You may edit DATA_PATH and NAME below.
"""

import subprocess
import time
import shlex
import os
from datetime import datetime
from pathlib import Path
import csv

# ==== User-configurable ====
DATA_PATH = "./data/ALLAML_10.npz"
NAME = "ALLAML"

# Fixed knobs per your request
IS_SNP = "0"
SEARCH_FLAGS = "--do_parameter_search 1 --n_trials 2 --n_jobs 2 --eval_metric loss"
NO_SEARCH_FLAGS = "--do_parameter_search 0"
EVAL_FLAGS = "--use_evaluation 1 --max_features 4 --feature_step 2 --max_features_graces 6"
NO_EVAL_FLAGS = "--use_evaluation 0 --max_features_graces 4"

# Methods
CORE_METHODS = ["EARFS","CANCELOUT","GRACES"]
INTERP_METHODS = ["FeatureAblation","Occlusion","Lime","DeepLIFT","GradientShap","LRP"]
OTHER_METHODS = ["BCOR","LASSO","FTEST","RF"]






# Build the 4 scenarios for a given method
# A: no search / no eval
# B: search / no eval
# C: no search / eval
# D: search / eval
def build_scenarios(method: str):
    base = f"python main.py --input_path {DATA_PATH} --name {NAME} --method {method} --is_snp {IS_SNP}"
    return [
        # ("A_noSearch_noEval", f"{base} {NO_SEARCH_FLAGS} {NO_EVAL_FLAGS}"),
        # ("B_search_noEval",   f"{base} {SEARCH_FLAGS} {NO_EVAL_FLAGS}"),
        # ("C_noSearch_Eval",   f"{base} {NO_SEARCH_FLAGS} {EVAL_FLAGS}"),
        ("D_search_Eval",     f"{base} {SEARCH_FLAGS} {EVAL_FLAGS}"),
    ]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    all_methods = CORE_METHODS + INTERP_METHODS + OTHER_METHODS
    runs = []
    for method in all_methods:
        runs.extend([(method, scen, cmd) for scen, cmd in build_scenarios(method)])

    logs_root = Path("run_logs")
    ensure_dir(logs_root)
    summary_csv = logs_root / "runs_summary.csv"

    print(f"[INFO] Total runs: {len(runs)}")
    print(f"[INFO] Logs directory: {logs_root.resolve()}")
    print(f"[INFO] Summary CSV: {summary_csv.resolve()}")
    print("-" * 80)

    # write CSV header
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index","method","scenario","returncode","start_time","end_time","duration_sec","command"])

    for idx, (method, scenario, cmd) in enumerate(runs, start=1):
        start_ts = time.time()
        start_iso = datetime.now().isoformat(timespec="seconds")

        # log files
        prefix = f"{idx:03d}_{method}_{scenario}"
        out_path = logs_root / f"{prefix}.out.txt"
        err_path = logs_root / f"{prefix}.err.txt"

        print(f"[RUN {idx:03d}] {method} | {scenario}")
        print(f"  CMD: {cmd}")
        try:
            with open(out_path, "w", encoding="utf-8") as fout, open(err_path, "w", encoding="utf-8") as ferr:
                # Use shell=False + shlex.split for safety
                proc = subprocess.run(shlex.split(cmd), stdout=fout, stderr=ferr, check=False)
                rc = proc.returncode
        except Exception as e:
            rc = -9999
            with open(err_path, "a", encoding="utf-8") as ferr:
                ferr.write(f"\n[EXCEPTION] {e}\n")

        end_ts = time.time()
        end_iso = datetime.now().isoformat(timespec="seconds")
        dur = round(end_ts - start_ts, 2)

        print(f"  -> returncode={rc}, duration={dur}s, out={out_path.name}, err={err_path.name}")
        if rc != 0:
            print(f"  !! FAILED, but continuing ...")

        with open(summary_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([idx, method, scenario, rc, start_iso, end_iso, dur, cmd])

        print("-" * 80)

    print("[DONE] All runs complete.")
    print(f"Summary CSV: {summary_csv.resolve()}")
    print(f"Per-run logs: {logs_root.resolve()}")

if __name__ == "__main__":
    main()
