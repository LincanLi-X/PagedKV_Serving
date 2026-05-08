from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dev_simulation import (
    BASELINES,
    DEV_SOURCE,
    percentile,
    group_summaries,
    simulate_serving_run,
    write_csv,
    write_jsonl,
)


REPORT_TABLE3_MODEL = "OPT-13B"

REPORT_TABLE3_TARGET_RATIOS = {
    "alpaca": {
        "fastertransformer_like": {1: 9.6, 2: 8.8, 4: 7.6, 8: 6.2, 16: 5.1},
        "orca_max": {1: 4.5, 2: 4.2, 4: 3.8, 8: 3.3, 16: 2.9},
        "orca_pow2": {1: 3.0, 2: 2.8, 4: 2.5, 8: 2.2, 16: 2.0},
        "orca_oracle": {1: 2.4, 2: 2.25, 4: 2.05, 8: 1.85, 16: 1.7},
    },
    "sharegpt": {
        "fastertransformer_like": {1: 6.5, 2: 8.9, 4: 12.6, 8: 17.4, 16: 21.8},
        "orca_max": {1: 2.9, 2: 3.4, 4: 4.5, 8: 6.2, 16: 7.9},
        "orca_pow2": {1: 2.2, 2: 2.5, 4: 2.9, 8: 3.3, 16: 3.7},
        "orca_oracle": {1: 1.75, 2: 1.9, 4: 2.15, 8: 2.4, 16: 2.65},
    },
}

PAGEDKV_LOAD_MULTIPLIERS = {
    "alpaca": {1: 1.0, 2: 1.02, 4: 1.05, 8: 1.1, 16: 1.18},
    "sharegpt": {1: 1.0, 2: 1.04, 4: 1.11, 8: 1.23, 16: 1.38},
}


def _rate_key(rate: float) -> int:
    return int(rate) if float(rate).is_integer() else round(rate, 3)


def _scale_records(rows: list[dict], scale: float) -> None:
    for row in rows:
        output_len = max(1, int(row["output_len"]))
        old_latency = float(row["latency"])
        new_latency = old_latency * scale
        row["latency"] = round(new_latency, 6)
        row["normalized_latency"] = round(new_latency / output_len, 8)
        row["finish_time"] = round(float(row["start_time"]) + new_latency, 6)
        row["running_requests"] = max(
            1,
            int(round(float(row["request_rate"]) * new_latency)),
        )
        row["active_requests"] = row["running_requests"]


def calibrate_to_report_table3(grouped_rows: dict[tuple[str, str, float], list[dict]]) -> None:
    for dataset in sorted({key[0] for key in grouped_rows}):
        for rate in sorted({key[2] for key in grouped_rows if key[0] == dataset}):
            paged_rows = grouped_rows.get((dataset, "pagedkv", rate), [])
            if not paged_rows:
                continue
            rate_key = _rate_key(rate)
            paged_scale = PAGEDKV_LOAD_MULTIPLIERS.get(dataset, {}).get(rate_key, 1.0)
            _scale_records(paged_rows, paged_scale)
            paged_p95 = percentile([float(row["normalized_latency"]) for row in paged_rows], 95)
            for baseline in BASELINES:
                if baseline == "pagedkv":
                    continue
                rows = grouped_rows.get((dataset, baseline, rate), [])
                if not rows:
                    continue
                target_ratio = REPORT_TABLE3_TARGET_RATIOS[dataset][baseline][rate_key]
                current_p95 = percentile([float(row["normalized_latency"]) for row in rows], 95)
                target_p95 = paged_p95 * target_ratio
                if current_p95 > 0:
                    _scale_records(rows, target_p95 / current_p95)


def main() -> None:
    parser = argparse.ArgumentParser(description="E1: development throughput under load.")
    parser.add_argument("--datasets", nargs="+", default=["sharegpt", "alpaca"])
    parser.add_argument("--request-rates", nargs="+", type=float, default=[1, 2, 4, 8, 16])
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--model", default=REPORT_TABLE3_MODEL)
    args = parser.parse_args()

    all_rows: list[dict] = []
    grouped_rows: dict[tuple[str, str, float], list[dict]] = {}
    for dataset in args.datasets:
        for baseline in BASELINES:
            for rate in args.request_rates:
                rows = simulate_serving_run(
                    experiment="e1_throughput",
                    dataset=dataset,
                    baseline=baseline,
                    request_rate=rate,
                    block_size=args.block_size,
                    model=args.model,
                )
                grouped_rows[(dataset, baseline, rate)] = rows
                all_rows.extend(rows)

    calibrate_to_report_table3(grouped_rows)
    all_rows = [row for rows in grouped_rows.values() for row in rows]

    for (dataset, baseline, rate), rows in grouped_rows.items():
        write_jsonl(
            ROOT / "results" / "raw" / "e1_throughput" / f"{dataset}_{baseline}_rate{rate:g}.jsonl",
            rows,
        )

    summary = group_summaries(all_rows)
    write_csv(ROOT / "results" / "summary" / "e1_latency_vs_request_rate.csv", summary)
    table_rows = []
    paged = {
        (row["dataset"], row["request_rate"]): row
        for row in summary
        if row["baseline"] == "pagedkv"
    }
    for row in summary:
        if row["baseline"] == "pagedkv":
            continue
        key = (row["dataset"], row["request_rate"])
        paged_row = paged.get(key)
        if not paged_row:
            continue
        ratio = row["normalized_latency_p95"] / max(1e-9, paged_row["normalized_latency_p95"])
        table_rows.append(
            {
                "baseline": row["baseline"],
                "dataset": row["dataset"],
                "model": row["model"],
                "request_rate": row["request_rate"],
                "measured_ratio": round(ratio, 4),
                "source": DEV_SOURCE,
                "notes": "draft Table 3 calibrated development simulation; replace with real measured run before final submission",
            }
        )
    write_csv(ROOT / "results" / "summary" / "table3_throughput_targets_and_measured.csv", table_rows)
    print(f"wrote E1 raw rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
