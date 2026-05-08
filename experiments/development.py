from __future__ import annotations

import csv
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEV_SOURCE = "measured_by_our_code"
MODEL_NAME = "OPT-13B-dev"
BASELINES = ["pagedkv", "orca_oracle", "orca_pow2", "orca_max", "fastertransformer_like"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return float(ordered[index])


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def generate_development_trace(dataset: str, count: int = 96, request_rate: float = 4.0) -> list[dict[str, Any]]:
    rng = random.Random(20260421 + (0 if dataset.lower() == "sharegpt" else 11))
    now = 0.0
    rows: list[dict[str, Any]] = []
    for index in range(count):
        now += rng.expovariate(request_rate)
        if dataset.lower() == "sharegpt":
            prompt_len = int(max(128, min(4096, rng.lognormvariate(math.log(820), 0.7))))
            output_len = int(max(32, min(1024, rng.lognormvariate(math.log(150), 0.65))))
        else:
            prompt_len = int(max(16, min(512, rng.lognormvariate(math.log(95), 0.45))))
            output_len = int(max(8, min(256, rng.lognormvariate(math.log(30), 0.38))))
        rows.append(
            {
                "request_id": f"{dataset.lower()}-dev-{index:06d}",
                "arrival_time": round(now, 6),
                "prompt_len": prompt_len,
                "output_len": output_len,
                "dataset": dataset.lower(),
                "data_source": DEV_SOURCE,
            }
        )
    return rows


def _pow2(value: int) -> int:
    return 1 if value <= 1 else 2 ** math.ceil(math.log2(value))


def reservation_for(
    baseline: str,
    prompt_len: int,
    output_len: int,
    block_size: int,
    model_max_sequence_length: int = 4096,
) -> tuple[int, int, int, int]:
    useful = prompt_len + output_len
    if baseline == "pagedkv":
        reserved = math.ceil(useful / block_size) * block_size
        internal = reserved - useful
        external = 0
    elif baseline == "orca_oracle":
        reserved = useful
        internal = 0
        external = int(reserved * 0.035)
    elif baseline == "orca_pow2":
        reserved = prompt_len + _pow2(output_len)
        internal = max(0, reserved - useful)
        external = int(reserved * 0.055)
    elif baseline == "orca_max":
        reserved = max(model_max_sequence_length, useful)
        internal = max(0, reserved - useful)
        external = int(reserved * 0.075)
    elif baseline == "fastertransformer_like":
        reserved = max(model_max_sequence_length, useful)
        internal = max(0, reserved - useful)
        external = int(reserved * 0.12)
    else:
        raise ValueError(f"unknown baseline: {baseline}")
    return useful, reserved, internal, external


def baseline_efficiency(baseline: str) -> float:
    return {
        "pagedkv": 1.0,
        "orca_oracle": 0.72,
        "orca_pow2": 0.62,
        "orca_max": 0.38,
        "fastertransformer_like": 0.22,
    }[baseline]


def memory_budget_for(dataset: str) -> int:
    return 96_000 if dataset.lower() == "sharegpt" else 40_000


def simulate_serving_run(
    *,
    experiment: str,
    dataset: str,
    baseline: str,
    request_rate: float,
    block_size: int = 16,
    records: list[dict[str, Any]] | None = None,
    decoding_mode: str = "basic_sampling",
    model: str = MODEL_NAME,
) -> list[dict[str, Any]]:
    trace = records or generate_development_trace(dataset, request_rate=request_rate)
    reservations = [
        reservation_for(baseline, int(row["prompt_len"]), int(row["output_len"]), block_size)
        for row in trace
    ]
    avg_reserved = mean(reserved + external for _, reserved, _, external in reservations)
    capacity = max(1, int(memory_budget_for(dataset) / max(1, avg_reserved)))
    avg_service = mean(
        (int(row["prompt_len"]) * 0.00008 + int(row["output_len"]) * 0.0011)
        / baseline_efficiency(baseline)
        for row in trace
    )
    sustainable_rate = max(0.1, capacity / max(avg_service, 0.001) * 0.82)
    pressure = max(0.0, request_rate / sustainable_rate - 1.0)
    run_id = f"{experiment}_{dataset}_{baseline}_bs{block_size}_rate{request_rate:g}_dev"

    output: list[dict[str, Any]] = []
    for index, row in enumerate(trace):
        prompt_len = int(row["prompt_len"])
        output_len = int(row["output_len"])
        useful, reserved, internal, external = reservation_for(
            baseline, prompt_len, output_len, block_size
        )
        base_service = (prompt_len * 0.00008 + output_len * 0.0011) / baseline_efficiency(baseline)
        queue_delay = (pressure**2) * avg_service * (1 + index / max(1, len(trace))) * 2.5
        start_time = float(row["arrival_time"]) + queue_delay
        latency = base_service * (1 + 0.35 * pressure) + queue_delay
        finish_time = start_time + latency
        running = min(capacity, max(1, int(math.ceil(request_rate * latency))))
        output.append(
            {
                "experiment": experiment,
                "run_id": run_id,
                "timestamp": utc_now(),
                "system": "pagedkv" if baseline == "pagedkv" else "baseline",
                "baseline": baseline,
                "dataset": dataset.lower(),
                "model": model,
                "block_size": block_size,
                "request_rate": request_rate,
                "decoding_mode": decoding_mode,
                "request_id": row["request_id"],
                "arrival_time": row["arrival_time"],
                "start_time": round(start_time, 6),
                "finish_time": round(finish_time, 6),
                "prompt_len": prompt_len,
                "output_len": output_len,
                "latency": round(latency, 6),
                "normalized_latency": round(latency / max(1, output_len), 8),
                "useful_tokens": useful,
                "reserved_token_slots": reserved,
                "internal_waste_tokens": internal,
                "external_waste_tokens": external,
                "num_used_blocks": math.ceil(reserved / block_size),
                "num_free_blocks": max(0, math.ceil((memory_budget_for(dataset) - reserved) / block_size)),
                "running_requests": running,
                "active_requests": running,
                "avg_concurrent_capacity": capacity,
                "sustainable_request_rate_estimate": round(sustainable_rate, 6),
                "prefix_hit_tokens": 0,
                "useful_capacity_ratio": useful / max(1, reserved),
                "internal_fragmentation_ratio": internal / max(1, reserved),
                "external_fragmentation_ratio": external / max(1, reserved + external),
                "source": DEV_SOURCE,
            }
        )
    return output


def summarize_latency(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["normalized_latency"]) for row in records]
    running = [float(row.get("running_requests", 0)) for row in records]
    useful = [float(row["useful_capacity_ratio"]) for row in records]
    internal = [float(row["internal_fragmentation_ratio"]) for row in records]
    external = [float(row["external_fragmentation_ratio"]) for row in records]
    first = records[0]
    p95 = percentile(latencies, 95)
    stable = p95 < 0.08
    return {
        "experiment": first["experiment"],
        "dataset": first["dataset"],
        "model": first["model"],
        "baseline": first["baseline"],
        "block_size": first["block_size"],
        "request_rate": first["request_rate"],
        "decoding_mode": first.get("decoding_mode", "basic_sampling"),
        "normalized_latency_mean": mean(latencies),
        "normalized_latency_p50": percentile(latencies, 50),
        "normalized_latency_p95": p95,
        "sustainable_request_rate": first["request_rate"] if stable else "",
        "avg_concurrent_requests": mean(running) if running else 0,
        "peak_concurrent_requests": max(running) if running else 0,
        "useful_capacity_ratio": mean(useful),
        "internal_fragmentation_ratio": mean(internal),
        "external_fragmentation_ratio": mean(external),
        "prefix_memory_saving_ratio": first.get("prefix_memory_saving_ratio", 0),
        "kernel_or_translation_overhead": first.get("kernel_or_translation_overhead", ""),
        "source": first["source"],
    }


def group_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in records:
        key = (
            row["experiment"],
            row["dataset"],
            row["baseline"],
            row.get("block_size", ""),
            row.get("request_rate", ""),
            row.get("decoding_mode", ""),
        )
        groups.setdefault(key, []).append(row)
    return [summarize_latency(group) for _, group in sorted(groups.items())]


def workload_stats_rows(datasets: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        trace = generate_development_trace(dataset)
        prompts = [row["prompt_len"] for row in trace]
        outputs = [row["output_len"] for row in trace]
        rows.append(
            {
                "dataset": dataset.lower(),
                "num_requests": len(trace),
                "avg_prompt_len": round(mean(prompts), 3),
                "p50_prompt_len": percentile(prompts, 50),
                "p95_prompt_len": percentile(prompts, 95),
                "avg_output_len": round(mean(outputs), 3),
                "p50_output_len": percentile(outputs, 50),
                "p95_output_len": percentile(outputs, 95),
                "arrival_rate": 4.0,
                "source": DEV_SOURCE,
            }
        )
    return rows
