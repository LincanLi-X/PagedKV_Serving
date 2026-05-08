from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dev_simulation import (
    group_summaries,
    simulate_serving_run,
    workload_stats_rows,
    write_csv,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="E3: development workload sensitivity.")
    parser.add_argument("--datasets", nargs="+", default=["sharegpt", "alpaca"])
    parser.add_argument("--baselines", nargs="+", default=["pagedkv", "orca_oracle", "orca_max"])
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--request-rate", type=float, default=4.0)
    args = parser.parse_args()

    all_rows = []
    for dataset in args.datasets:
        for baseline in args.baselines:
            rows = simulate_serving_run(
                experiment="e3_workload_sensitivity",
                dataset=dataset,
                baseline=baseline,
                request_rate=args.request_rate,
                block_size=args.block_size,
            )
            all_rows.extend(rows)
            write_jsonl(
                ROOT / "results" / "raw" / "e3_workload_sensitivity" / f"{dataset}_{baseline}.jsonl",
                rows,
            )

    write_csv(ROOT / "results" / "summary" / "e3_workload_sensitivity.csv", group_summaries(all_rows))
    write_csv(ROOT / "results" / "summary" / "workload_stats.csv", workload_stats_rows(args.datasets))
    print(f"wrote E3 raw rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
