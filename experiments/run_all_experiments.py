from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run currently implemented Step 0-8 reproducibility tasks.")
    parser.add_argument("--profile", default="course_cpu")
    args = parser.parse_args()
    print(f"Running implemented setup tasks for profile={args.profile}", flush=True)
    trace_cmd = [
        sys.executable,
        "experiments/workloads/prepare_workload_traces.py",
        "--output-dir",
        "experiments/workloads",
        "--summary",
        "results/summary/workload_stats.csv",
    ]
    sharegpt_input = os.environ.get("SHAREGPT_INPUT")
    alpaca_input = os.environ.get("ALPACA_INPUT")
    if sharegpt_input:
        trace_cmd.extend(["--sharegpt-input", sharegpt_input])
    if alpaca_input:
        trace_cmd.extend(["--alpaca-input", alpaca_input])
    if sharegpt_input or alpaca_input:
        subprocess.run(trace_cmd, cwd=ROOT, check=True)
    else:
        print("No real workload input provided; skipping trace preparation.", flush=True)
    for script in [
        "experiments/run_e1_throughput.py",
        "experiments/run_e2_memory_efficiency.py",
        "experiments/run_e3_workload_sensitivity.py",
        "experiments/run_e4_prefix_sharing.py",
        "experiments/run_e5_block_size_ablation.py",
    ]:
        subprocess.run([sys.executable, script], cwd=ROOT, check=True)

if __name__ == "__main__":
    main()
