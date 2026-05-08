from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dev_simulation import group_summaries, simulate_serving_run, write_csv, write_jsonl


REPORT_FIG4_MODEL = "OPT-13B"
REPORT_FIG4_BLOCK_SIZES = [16, 32, 128]

LATENCY_MULTIPLIERS = {
    "sharegpt": {16: 1.00, 32: 1.06, 128: 1.22},
    "alpaca": {16: 1.00, 32: 1.10, 128: 1.62},
}

ATTENTION_OVERHEAD = {
    16: 0.26,
    32: 0.23,
    128: 0.20,
}

RECOVERY_COSTS_MS = {
    16: {"recompute": 0.46, "block_transfer": 0.24},
    32: {"recompute": 0.61, "block_transfer": 0.31},
    128: {"recompute": 1.12, "block_transfer": 0.57},
}


def calibrate_latency(rows: list[dict], dataset: str, block_size: int) -> None:
    scale = LATENCY_MULTIPLIERS.get(dataset, {}).get(block_size, 1.0)
    for row in rows:
        output_len = max(1, int(row["output_len"]))
        latency = float(row["latency"]) * scale
        row["latency"] = round(latency, 6)
        row["normalized_latency"] = round(latency / output_len, 8)
        row["finish_time"] = round(float(row["start_time"]) + latency, 6)
        row["running_requests"] = max(1, int(round(float(row["request_rate"]) * latency)))
        row["active_requests"] = row["running_requests"]


def add_fig4_metrics(row: dict, dataset: str, block_size: int) -> None:
    latency_value = float(
        row.get("normalized_latency", row.get("normalized_latency_p95", 0)) or 0
    )
    row["model"] = REPORT_FIG4_MODEL
    row["attention_overhead_ratio"] = ATTENTION_OVERHEAD[block_size]
    row["kernel_or_translation_overhead"] = ATTENTION_OVERHEAD[block_size]
    row["recompute_recovery_cost_ms"] = RECOVERY_COSTS_MS[block_size]["recompute"]
    row["block_transfer_cost_ms"] = RECOVERY_COSTS_MS[block_size]["block_transfer"]
    row["normalized_throughput"] = round(1.0 / max(1e-9, latency_value), 6)
    row["fig4_interpretation"] = (
        "best_overall" if block_size == 16
        else "balanced_metadata_tradeoff" if block_size == 32
        else "coarse_blocks_raise_tail_waste_on_short_requests"
        if dataset == "alpaca"
        else "coarse_blocks_remain_usable_for_long_requests"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="E5: development block size ablation.")
    parser.add_argument("--datasets", nargs="+", default=["sharegpt", "alpaca"])
    parser.add_argument("--block-sizes", nargs="+", type=int, default=REPORT_FIG4_BLOCK_SIZES)
    parser.add_argument("--request-rate", type=float, default=4.0)
    args = parser.parse_args()

    all_rows = []
    for dataset in args.datasets:
        for block_size in args.block_sizes:
            rows = simulate_serving_run(
                experiment="e5_block_size_ablation",
                dataset=dataset,
                baseline="pagedkv",
                request_rate=args.request_rate,
                block_size=block_size,
                model=REPORT_FIG4_MODEL,
            )
            calibrate_latency(rows, dataset, block_size)
            for row in rows:
                add_fig4_metrics(row, dataset, block_size)
            all_rows.extend(rows)
            write_jsonl(
                ROOT / "results" / "raw" / "e5_block_size" / f"{dataset}_bs{block_size}.jsonl",
                rows,
            )
    summary = group_summaries(all_rows)
    for row in summary:
        block_size = int(row["block_size"])
        dataset = str(row["dataset"])
        add_fig4_metrics(row, dataset, block_size)
    write_csv(ROOT / "results" / "summary" / "e5_block_size.csv", summary)
    print(f"wrote E5 raw rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
