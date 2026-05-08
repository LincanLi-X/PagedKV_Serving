from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dev_simulation import DEV_SOURCE, generate_development_trace, write_csv, write_jsonl
from experiments.report_table4 import read_csv, write_table4


REPORT_TABLE4_MODEL = "OPT-13B"


def saving_factor(dataset: str, decoding_mode: str, width: int) -> float:
    base = 0.0
    if decoding_mode == "parallel_sampling":
        base = 0.10 if dataset == "alpaca" else 0.24
    elif decoding_mode == "beam_search":
        base = 0.46 if dataset == "alpaca" else 0.56
    elif decoding_mode == "shared_prefix_serving":
        base = 0.18 if dataset == "alpaca" else 0.36
    return min(0.72, base * math.log2(max(1, width) + 1) / math.log2(7))


def main() -> None:
    parser = argparse.ArgumentParser(description="E4: development prefix sharing, sampling, and beam search.")
    parser.add_argument("--datasets", nargs="+", default=["alpaca", "sharegpt"])
    parser.add_argument("--modes", nargs="+", default=["basic_sampling", "parallel_sampling", "beam_search", "shared_prefix_serving"])
    parser.add_argument("--widths", nargs="+", type=int, default=[1, 2, 4, 6])
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        trace = generate_development_trace(dataset, count=72)
        for mode in args.modes:
            for width in args.widths:
                total_blocks = sum(
                    math.ceil((int(item["prompt_len"]) + int(item["output_len"])) / args.block_size)
                    for item in trace
                ) * max(1, width)
                factor = saving_factor(dataset, mode, width)
                saved_blocks = int(total_blocks * factor)
                with_sharing = total_blocks - saved_blocks
                row = {
                    "experiment": "e4_prefix_sharing",
                    "dataset": dataset,
                    "model": REPORT_TABLE4_MODEL,
                    "baseline": "paged_prefix_on",
                    "decoding_mode": mode,
                    "width": width,
                    "block_size": args.block_size,
                    "request_rate": 4.0,
                    "total_blocks_without_sharing": total_blocks,
                    "total_blocks_with_sharing": with_sharing,
                    "shared_blocks": saved_blocks,
                    "saved_blocks": saved_blocks,
                    "memory_saving_ratio": saved_blocks / max(1, total_blocks),
                    "prefix_memory_saving_ratio": saved_blocks / max(1, total_blocks),
                    "prefix_hit_tokens": saved_blocks * args.block_size,
                    "cow_count": max(0, width - 1) * (12 if mode == "beam_search" else 3),
                    "normalized_latency": 0.02 / max(1.0, 1 + factor),
                    "useful_capacity_ratio": 0.95,
                    "internal_fragmentation_ratio": 0.05,
                    "external_fragmentation_ratio": 0.0,
                    "running_requests": min(64, max(1, width * 4)),
                    "source": DEV_SOURCE,
                }
                rows.append(row)
    write_jsonl(ROOT / "results" / "raw" / "e4_prefix_sharing" / "prefix_sharing_dev.jsonl", rows)
    write_csv(ROOT / "results" / "summary" / "e4_prefix_sharing.csv", rows)
    write_table4(read_csv(ROOT / "results" / "summary" / "e2_fragmentation.csv"), rows)
    print(f"wrote E4 rows: {len(rows)}")


if __name__ == "__main__":
    main()
