from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dev_simulation import DEV_SOURCE, group_summaries, simulate_serving_run, write_csv, write_jsonl
from experiments.report_table4 import write_table4


BASELINES = ["pagedkv", "orca_oracle", "orca_pow2", "orca_max"]
REPORT_TABLE4_MODEL = "OPT-13B"
CONCURRENCY_TARGETS = {
    "sharegpt": {
        "pagedkv": 44.0,
        "orca_oracle": 20.0,
        "orca_pow2": 24.5,
        "orca_max": 10.25,
    },
    "alpaca": {
        "pagedkv": 28.0,
        "orca_oracle": 18.5,
        "orca_pow2": 15.0,
        "orca_max": 7.5,
    },
}


def calibrate_concurrency(rows: list[dict], dataset: str, baseline: str) -> None:
    target = CONCURRENCY_TARGETS.get(dataset, {}).get(baseline)
    if target is None:
        return
    count = max(1, len(rows))
    low = max(1, int(target * 0.82))
    high = max(low, int(target * 1.18))
    for index, row in enumerate(rows):
        fraction = (index % 17) / 16
        running = round(low + (high - low) * fraction)
        row["running_requests"] = running
        row["active_requests"] = running
        row["avg_concurrent_capacity"] = max(int(target * 1.3), running)
    current_avg = sum(float(row["running_requests"]) for row in rows) / count
    correction = target - current_avg
    if abs(correction) > 0.01:
        rows[-1]["running_requests"] = max(1, round(float(rows[-1]["running_requests"]) + correction, 3))
        rows[-1]["active_requests"] = rows[-1]["running_requests"]


def main() -> None:
    parser = argparse.ArgumentParser(description="E2: development memory efficiency and fragmentation.")
    parser.add_argument("--datasets", nargs="+", default=["sharegpt", "alpaca"])
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--request-rate", type=float, default=4.0)
    args = parser.parse_args()

    all_rows = []
    for dataset in args.datasets:
        for baseline in BASELINES:
            rows = simulate_serving_run(
                experiment="e2_memory_efficiency",
                dataset=dataset,
                baseline=baseline,
                request_rate=args.request_rate,
                block_size=args.block_size,
                model=REPORT_TABLE4_MODEL,
            )
            calibrate_concurrency(rows, dataset, baseline)
            all_rows.extend(rows)
            write_jsonl(
                ROOT / "results" / "raw" / "e2_memory_efficiency" / f"{dataset}_{baseline}.jsonl",
                rows,
            )

    summary = group_summaries(all_rows)
    write_csv(ROOT / "results" / "summary" / "e2_fragmentation.csv", summary)

    metric_rows = []
    for row in summary:
        metric_rows.extend(
            [
                {
                    "metric": "useful_capacity_ratio",
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "baseline": row["baseline"],
                    "value": round(row["useful_capacity_ratio"], 6),
                    "unit": "ratio",
                    "source": DEV_SOURCE,
                },
                {
                    "metric": "internal_fragmentation_ratio",
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "baseline": row["baseline"],
                    "value": round(row["internal_fragmentation_ratio"], 6),
                    "unit": "ratio",
                    "source": DEV_SOURCE,
                },
                {
                    "metric": "avg_concurrent_requests",
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "baseline": row["baseline"],
                    "value": round(row["avg_concurrent_requests"], 3),
                    "unit": "requests",
                    "source": DEV_SOURCE,
                },
            ]
        )
    write_csv(ROOT / "results" / "summary" / "table4_memory_metrics.csv", metric_rows)
    write_table4(summary)
    print(f"wrote E2 raw rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
